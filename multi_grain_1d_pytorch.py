import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import shutil 
from scipy.optimize import curve_fit
from scipy.integrate import solve_bvp

# --- Set PyTorch Device and Dtype ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
print(f"Running Phase-Field Simulation on: {device.type.upper()} with {dtype}")

# =============================================================================
# 0. ANALYTICAL BVP & DEBYE UTILS (Kept in SciPy/NumPy for curve-fitting)
# =============================================================================

def calculate_gouy_chapman_profile(x_nm, center_nm, phi_max, T, eps_r, c_dop_bulk, N_vo, N_dop, z_vo, z_dop):
    R, F, eps0, e_charge = 8.314, 96485.0, 8.854e-12, 1.6022e-19
    Vt = (R * T) / F
    c_vo_bulk = -(z_dop * c_dop_bulk) / z_vo
    
    def get_concentrations(phi):
        term_vo = ((N_vo / c_vo_bulk) - 1.0) * np.exp(z_vo * phi / Vt)
        c_vo = N_vo / (1.0 + term_vo)
        term_dop = ((N_dop / c_dop_bulk) - 1.0) * np.exp(z_dop * phi / Vt)
        c_dop = N_dop / (1.0 + term_dop)
        return c_vo, c_dop

    def ode_system(x_nano, y):
        phi, dphi_dx_nano = y[0], y[1] 
        c_vo, c_dop = get_concentrations(phi)
        rho = e_charge * (z_vo * c_vo + z_dop * c_dop)
        eps = eps0 * eps_r
        d2phi_dx2_nano = (-rho / eps) * (1e-9)**2 
        return np.vstack((dphi_dx_nano, d2phi_dx2_nano))

    def boundary_conditions(ya, yb):
        return np.array([ya[0] - phi_max, yb[0] - 0.0])

    max_dist_nm = np.max(np.abs(x_nm - center_nm))
    x_bvp = np.linspace(0, max_dist_nm, 1000)
    
    L_guess_nm = 2.0  
    y_guess = np.zeros((2, x_bvp.size))
    y_guess[0] = phi_max * np.exp(-x_bvp / L_guess_nm)
    y_guess[1] = -(phi_max / L_guess_nm) * np.exp(-x_bvp / L_guess_nm)

    res = solve_bvp(ode_system, boundary_conditions, x_bvp, y_guess, tol=1e-5, max_nodes=50000)
    
    if not res.success:
        print(f"Warning: Gouy-Chapman BVP failed! Reason: {res.message}")
        return np.zeros_like(x_nm)

    dist_grid_nm = np.abs(x_nm - center_nm)
    return res.sol(dist_grid_nm)[0]

def exponential_decay(x, A, L, y0):
    return A * np.exp(-x / L) + y0

def fit_debye_length(x_nm, phi_phys):
    gb_idx = np.argmax(np.abs(phi_phys))
    x_gb = x_nm[gb_idx]
    offset_idx = gb_idx + 10 
    
    x_fit = x_nm[offset_idx:] - x_gb 
    phi_fit = phi_phys[offset_idx:]
    p0 = [phi_fit[0] - phi_phys[0], 1.0, phi_phys[0]]
    
    try:
        popt, _ = curve_fit(exponential_decay, x_fit, phi_fit, p0=p0)
        return popt[1]
    except RuntimeError:
        return None
    
def debye_length(eps_r, T, c_dop_inf, kb=1.3806e-23, eps0=8.854e-12, e_charge=1.6022e-19):
    return np.sqrt((eps0 * eps_r * kb * T) / (2 * (e_charge**2) * c_dop_inf)) 

def extract_debye_length_1e(x_nm, phi_phys):
    gb_idx = np.argmax(np.abs(phi_phys))
    x_gb, phi_gb, phi_bulk = x_nm[gb_idx], phi_phys[gb_idx], phi_phys[0] 
    
    threshold_phi = phi_bulk + (phi_gb - phi_bulk) * np.exp(-1) 
    right_side_x, right_side_phi = x_nm[gb_idx:], phi_phys[gb_idx:]
    x_threshold = np.interp(threshold_phi, right_side_phi[::-1], right_side_x[::-1])
    
    return abs(x_threshold - x_gb)

# =============================================================================
# 1. 1D SOLVER UTILITIES (PyTorch)
# =============================================================================

def laplacian_neumann_1d(field, dx):
    """Computes 1D Laplacian for a 2D array (Nx, N_grains) with Neumann BCs."""
    lap = torch.zeros_like(field)
    lap[1:-1] = (field[:-2] - 2.0 * field[1:-1] + field[2:]) / (dx**2)
    lap[0] = (2.0 * field[1] - 2.0 * field[0]) / (dx**2)
    lap[-1] = (2.0 * field[-2] - 2.0 * field[-1]) / (dx**2)
    return lap

def build_poisson_lu(Nx, dx_dimless):
    """Pre-computes the LU factorization of the Poisson matrix."""
    L = torch.zeros((Nx, Nx), dtype=dtype, device=device)
    main_diag = -2.0 / (dx_dimless**2)
    off_diag = 1.0 / (dx_dimless**2)
    
    idx = torch.arange(Nx, device=device)
    L[idx, idx] = main_diag
    L[idx[:-1], idx[1:]] = off_diag
    L[idx[1:], idx[:-1]] = off_diag
    
    # Boundary Conditions
    L[0, :] = 0.0; L[0, 0] = 1.0
    L[-1, :] = 0.0; L[-1, -1] = 1.0 / dx_dimless; L[-1, -2] = -1.0 / dx_dimless
    
    LU, pivots = torch.linalg.lu_factor(L)
    return LU, pivots

def solve_poisson_1d_fast(LU, pivots, C_Vo, C_dop, chi, z_Vo, z_dop):
    """Solves 1D Poisson using pre-computed LU factorization."""
    rho_dimless = z_Vo * C_Vo + z_dop * C_dop
    b = -chi * rho_dimless
    b[0] = 0.0
    b[-1] = 0.0
    
    # torch.linalg.lu_solve expects right-hand side to be 2D (Nx, 1)
    phi = torch.linalg.lu_solve(LU, pivots, b.unsqueeze(1)).squeeze(1)
    return phi

def initial_bicrystal_1d(Lx, Nx, num_grains=2):
    eta = torch.zeros((Nx, num_grains), dtype=dtype, device=device)
    x = torch.linspace(0, Lx, Nx, dtype=dtype, device=device)
    center = Lx / 2.0
    width_init = 5.0  
    
    eta[:, 0] = 0.5 * (1.0 - torch.tanh((x - center)/(width_init/2.0)))
    eta[:, 1] = 0.5 * (1.0 + torch.tanh((x - center)/(width_init/2.0)))
    return eta, x

def initial_defect_fields_1d(Nx, c_dop_bulk, z_dop, z_vo):
    C_dop = torch.ones(Nx, dtype=dtype, device=device) * c_dop_bulk
    ratio = - z_dop / z_vo
    C_vo = C_dop * ratio
    phi = torch.zeros(Nx, dtype=dtype, device=device)
    return C_dop, C_vo, phi

# =============================================================================
# 2. THERMODYNAMICS & CONCENTRATIONS (PyTorch)
# =============================================================================

def concentration_bulk_core(delta_g0, C, Nb, Nc, h, epsilon=1e-6):
    k = np.exp(delta_g0) # Scalar, so np is fine
    Cb = torch.zeros_like(C)
    Cc = torch.zeros_like(C)
    
    mask_bulk = h < epsilon
    mask_core = h > (1.0 - epsilon)
    mask_inter = ~(mask_bulk | mask_core)
    
    if torch.any(mask_bulk):
        Cb[mask_bulk] = C[mask_bulk]
        Cc[mask_bulk] = (C[mask_bulk] * Nc) / (k * Nb + C[mask_bulk] * (1.0 - k) + 1e-12)

    if torch.any(mask_core):
        Cc[mask_core] = C[mask_core]
        Cb[mask_core] = (k * C[mask_core] * Nb) / (Nc + C[mask_core] * (k - 1.0) + 1e-12)

    if torch.any(mask_inter):
        h_loc = h[mask_inter]
        C_loc = C[mask_inter]
        
        A = h_loc * (k - 1.0)
        B = h_loc * Nc + k * Nb * (1.0 - h_loc) - C_loc * (k - 1.0)
        D = -C_loc * Nc
        
        linear_idx = torch.abs(A) < 1e-12
        quad_idx = ~linear_idx
        Cc_res = torch.empty_like(C_loc)
        
        if torch.any(linear_idx):
            Cc_res[linear_idx] = -D[linear_idx] / (B[linear_idx] + 1e-12)
        if torch.any(quad_idx):
            disc = B[quad_idx]**2 - 4 * A[quad_idx] * D[quad_idx]
            Cc_res[quad_idx] = (-B[quad_idx] + torch.sqrt(torch.clamp(disc, min=0.0))) / (2 * A[quad_idx])
            
        Cc_res = torch.clamp(Cc_res, min=0.0, max=Nc)
        Cc[mask_inter] = Cc_res
        Cb[mask_inter] = (C_loc - h_loc * Cc_res) / (1.0 - h_loc)

    return Cb, Cc

def compute_chemical_potential(C_b, N_b, z, phi, g0=0.0):
    C_safe = torch.clamp(C_b, min=1e-12, max=N_b - 1e-12)
    return g0 + torch.log(C_safe / (N_b - C_safe)) + z * phi

def compute_kks_mobility(C_b, C_c, h, N_b, N_c, D_diff):
    inv_stiffness_b = (1.0 - (C_b / N_b)) * C_b
    inv_stiffness_c = (1.0 - (C_c / N_c)) * C_c
    inv_stiffness_eff = (1.0 - h) * inv_stiffness_b + h * inv_stiffness_c
    return D_diff * inv_stiffness_eff

def update_concentration_1d(C, mu, M, dt, dx):
    dmu_dx = (mu[1:] - mu[:-1]) / dx
    M_mid = 0.5 * (M[1:] + M[:-1])
    J_x = -M_mid * dmu_dx
    
    minus_div_J = torch.zeros_like(C)
    minus_div_J[0] -= J_x[0] / dx
    minus_div_J[1:-1] += (J_x[:-1] - J_x[1:]) / dx
    minus_div_J[-1] += J_x[-1] / dx

    return C + dt * minus_div_J

# =============================================================================
# 3. PHASE FIELD & ENERGY (PyTorch)
# =============================================================================

def compute_h_and_derivative_1d(eta):
    sum_eta3 = torch.sum(eta**3, dim=1, keepdim=True)
    sum_eta2 = torch.sum(eta**2, dim=1, keepdim=True)
    
    h = (4.0 / 3.0) * (1.0 - 4.0 * sum_eta3 + 3.0 * (sum_eta2**2))
    h = torch.squeeze(h, dim=1)  
    dh_deta = 16.0 * eta * (sum_eta2 - eta) 
    return h, dh_deta

def driving_force_concentrations(C_b, C_c, N_b, N_c):
    return N_b * torch.log(1 - C_b / N_b) - N_c * torch.log(1 - C_c / N_c)

def torch_gradient_1d(y, dx):
    """PyTorch equivalent of np.gradient for 1D arrays."""
    grad = torch.zeros_like(y)
    grad[0] = (y[1] - y[0]) / dx
    grad[-1] = (y[-1] - y[-2]) / dx
    grad[1:-1] = (y[2:] - y[:-2]) / (2.0 * dx)
    return grad

def compute_properties_1d(eta, dx, omega, gamma, kappa):
    eta0, eta1 = eta[:, 0], eta[:, 1]
    
    f_bulk_0 = eta0**4/4.0 - eta0**2/2.0
    f_bulk_1 = eta1**4/4.0 - eta1**2/2.0
    f_inter = gamma * (eta0**2) * (eta1**2) + 0.25
    f_local = (f_bulk_0 + f_bulk_1 + f_inter) * omega

    grad_eta0 = torch_gradient_1d(eta0, dx)
    grad_eta1 = torch_gradient_1d(eta1, dx)
    f_grad = 0.5 * kappa * (grad_eta0**2 + grad_eta1**2)
    
    total_energy_density = f_local + f_grad
    sigma_calculated = torch.sum(total_energy_density) * dx
    
    max_slope = torch.max(torch.abs(grad_eta0))
    width_calculated = 1.0 / max_slope if max_slope > 1e-6 else 0.0
    
    return sigma_calculated, width_calculated, total_energy_density, eta0, eta1

# =============================================================================
# 4. PLOTTING
# =============================================================================

def plot_fields_1d(output_dir, n, x_nm, phi_phys, rho_phys, C_vo_phys, C_dop_phys, eta0, eta1, E_dens_phys, sig_phys):
    fig, ax = plt.subplots(1, 5, figsize=(25, 4))
    
    ax[0].plot(x_nm, eta0, 'r-', label='Grain 1')
    ax[0].plot(x_nm, eta1, 'b--', label='Grain 2')
    ax[0].set_title(f"Order Parameters (Step {n})")
    ax[0].set_ylabel(r"$\eta$")
    ax[0].legend()
    
    ax[1].plot(x_nm, phi_phys, 'k-', lw=2)
    ax[1].set_title("Potential $\phi$ (V)")
    ax[1].set_ylabel("Volts")
    
    ax[2].plot(x_nm, rho_phys, 'r-', lw=2)
    ax[2].axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax[2].set_title(r"Charge Density $\rho$")
    ax[2].set_ylabel("$C/m^3$")
    
    ax[3].plot(x_nm, C_vo_phys, 'g-', lw=2)
    ax[3].set_yscale('log')
    ax[3].set_title(r"$C_{V_O}$ Profile")
    ax[3].set_ylabel("$m^{-3}$")
    
    ax[4].plot(x_nm, C_dop_phys, 'm-', lw=2)
    ax[4].set_yscale('log')
    ax[4].set_title(r"$C_{dop}$ Profile")
    ax[4].set_ylabel("$m^{-3}$")
    
    for a in ax:
        a.set_xlabel("Distance (nm)")
        a.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fields_1d_step_{n:05d}.png")
    plt.close(fig)

# =============================================================================
# 5. MAIN SIMULATION LOOP
# =============================================================================

def run_bicrystal_1d(output_dir, delta_g_dop_eV, delta_g_vo_eV=-1.5, N_vo_c_phys=1.0e27, N_dop_c_phys=1.68e26,
                                                            N_vo_b_phys=5.1e28, N_dop_b_phys=1.68e28, c_dop_bulk_phys=7.0e25):
    # Setup
    Lx, T = 80e-9, 1623.0 
    R, a, Na, w_c, sigma = 8.314, 3.90e-10, 6.02214076e23, 0.78e-9, 1.0
    V_m = Na * (a**3)
    
    kappa_phys = (3.0 / 4.0) * sigma * w_c
    omega_phys = 6.0 * sigma / w_c
    D_vo_phys, D_dop_phys = 1.35e-9, 1.3e-11 
    eps0, eps_r, F = 8.854e-12, 56.0, 96485.0
    
    z_vo, z_dop = 2.0, -1.0
    c_vo_bulk_phys = -(z_dop * c_dop_bulk_phys) / z_vo 

    # Scaling
    length_scale = a
    energy_scale = (R * T) / V_m    
    time_scale   = (a**2) / D_vo_phys
    phi_scale    = (R * T) / F
    conc_scale   = 1.0 / (a**3)

    Lx_dl = Lx / length_scale
    kappa = kappa_phys / (length_scale**2 * energy_scale)
    omega = omega_phys / energy_scale
    chi_poisson = (length_scale**2 * F * F) / (V_m * eps0 * eps_r * R * T)

    delta_g_vo = (delta_g_vo_eV * F) / (R * T)
    delta_g_dop = (delta_g_dop_eV * F) / (R * T)
    
    N_vo_b, N_dop_b = N_vo_b_phys * a**3, N_dop_b_phys * a**3
    N_vo_c, N_dop_c = N_vo_c_phys * a**3, N_dop_c_phys * a**3
    c_dop_init, c_vo_init = c_dop_bulk_phys * a**3, c_vo_bulk_phys * a**3

    D_vo_dim = 1.0
    # D_dop_switch = 1.35e-9 / 1.3e-11 
    D_dop_dim = (D_dop_phys / D_vo_phys) #* D_dop_switch
    L_mobility = 1.0 
    gamma = 1.5

    Nx = 1024
    dx = Lx_dl / Nx
    dt = (dx**2) / (4.0 * L_mobility * kappa)

    print(f"--- Setup 1D Benchmark (PyTorch) ---")
    print(f"Grid: {Nx}, dx={dx:.2f}, dt={dt:.4e}")

    # Initialize Tensors
    eta, x_dl = initial_bicrystal_1d(Lx_dl, Nx)
    C_dop, C_vo, phi = initial_defect_fields_1d(Nx, c_dop_init, z_dop, z_vo)
    
    # Pre-factorize Poisson Matrix (runs once!)
    LU_poisson, pivots_poisson = build_poisson_lu(Nx, dx)
    
    x_nm = (x_dl * length_scale * 1e9).cpu().numpy()

    steps = 500001
    nt = 10000

    h5_path = os.path.join(output_dir, f"results_data.h5")
    h5file = h5py.File(h5_path, "w")

    for n in range(steps):

        # KINETIC STARVATION FIX
        if n == 25000:
            L_mobility = 0.0
            dt = 0.45 * (dx**2) / max(D_vo_dim, D_dop_dim)
            print(f"\n>>> Phase-field frozen. Boosting dt to {dt:.4e} to accelerate diffusion <<<\n")
        
        h, dh_deta = compute_h_and_derivative_1d(eta)
        
        C_vo_b, C_vo_c = concentration_bulk_core(delta_g_vo, C_vo, N_vo_b, N_vo_c, h)
        C_dop_b, C_dop_c = concentration_bulk_core(delta_g_dop, C_dop, N_dop_b, N_dop_c, h)
        
        phi = solve_poisson_1d_fast(LU_poisson, pivots_poisson, C_vo, C_dop, chi_poisson, z_vo, z_dop)
        
        mu_vo = compute_chemical_potential(C_vo_b, N_vo_b, z_vo, phi, 0.0)
        mu_dop = compute_chemical_potential(C_dop_b, N_dop_b, z_dop, phi, 0.0)
        
        M_vo = compute_kks_mobility(C_vo_b, C_vo_c, h, N_vo_b, N_vo_c, D_vo_dim)
        M_dop = compute_kks_mobility(C_dop_b, C_dop_c, h, N_dop_b, N_dop_c, D_dop_dim)
        
        C_vo = update_concentration_1d(C_vo, mu_vo, M_vo, dt, dx)
        C_dop = update_concentration_1d(C_dop, mu_dop, M_dop, dt, dx)
        
        if L_mobility > 0.0:
            omega_diff_vo = driving_force_concentrations(C_vo_b, C_vo_c, N_vo_b, N_vo_c)
            omega_diff_dop = driving_force_concentrations(C_dop_b, C_dop_c, N_dop_b, N_dop_c)
            
            driving_force_scalar = omega_diff_vo + omega_diff_dop
            F_coupling = dh_deta * driving_force_scalar.unsqueeze(1)
            
            sum_sq = torch.sum(eta**2, dim=1, keepdim=True)
            df_bulk = omega * (eta**3 - eta)
            df_inter = 2.0 * omega * gamma * eta * (sum_sq - eta**2)
            
            lap = laplacian_neumann_1d(eta, dx)
            rhs = -L_mobility * (df_bulk + df_inter + F_coupling - kappa * lap)
            
            eta = eta + dt * rhs
        
        if n % nt == 0:
            # Transfer physics to CPU NumPy for logging/plotting
            sig_dim, wid_dim, E_dens_dim, eta0, eta1 = compute_properties_1d(eta, dx, omega, gamma, kappa)
            
            sig_phys_val = (sig_dim * energy_scale * length_scale).item()
            wid_phys_val = (wid_dim * length_scale).item()
            E_dens_phys = (E_dens_dim * energy_scale).cpu().numpy()
            phi_phys = (phi * phi_scale).cpu().numpy()
            C_vo_phys = (C_vo * conc_scale).cpu().numpy()
            C_dop_phys = (C_dop * conc_scale).cpu().numpy()
            rho_phys = ((z_vo * C_vo + z_dop * C_dop) * conc_scale * 1.6022e-19).cpu().numpy()
            eta0_np = eta0.cpu().numpy()
            eta1_np = eta1.cpu().numpy()

            print(f"\nStep {n}: \nMax Phi={np.max(phi_phys):.4f} V, Max Vo={np.max(C_vo_phys):.4e} 1/m3 | Sig={sig_phys_val:.4f} J/m^2 | Width={wid_phys_val*1e9:.4f} nm")

            if n > 20000:
                l_D_thoery = debye_length(eps_r, T, c_dop_inf=c_dop_bulk_phys)
                l_D_sim = fit_debye_length(x_nm, phi_phys)
                l_D_sim_a = extract_debye_length_1e(x_nm, phi_phys)
                if l_D_sim is not None:
                    print(f"Debye lengths: theory={l_D_thoery*1e9:.4f} nm & Simulation (method B)={l_D_sim:.4f} nm | method a={l_D_sim_a:.4f} nm")

            plot_fields_1d(output_dir, n, x_nm, phi_phys, rho_phys, C_vo_phys, C_dop_phys, eta0_np, eta1_np, E_dens_phys, sig_phys_val)

            grp = h5file.create_group(f"step_{n}") 
            grp["x_nm"] = x_nm 
            grp["eta"] = eta.cpu().numpy() 
            grp["phi_phys"] = phi_phys 
            grp["C_vo_phys"] = C_vo_phys 
            grp["C_dop_phys"] = C_dop_phys 
            grp["rho_phys"] = rho_phys
            
    h5file.close()

    # Final Gouy-Chapman Benchmark Calculation
    phi_phys = (phi * phi_scale).cpu().numpy()
    C_vo_phys = (C_vo * conc_scale).cpu().numpy()
    C_dop_phys = (C_dop * conc_scale).cpu().numpy()

    center_idx = np.argmax(np.abs(phi_phys))
    center_nm = x_nm[center_idx]
    phi_max = phi_phys[center_idx]

    phi_gc_theory = calculate_gouy_chapman_profile(
        x_nm=x_nm, center_nm=center_nm, phi_max=phi_max, T=1623.0, eps_r=56.0,
        c_dop_bulk=c_dop_bulk_phys, N_vo=N_vo_b_phys, N_dop=N_dop_b_phys, z_vo=2.0, z_dop=-1.0
    )

    plt.figure(figsize=(8, 5))
    plt.plot(x_nm, phi_phys, 'k-', lw=2.5, label='Phase-Field Simulation (Diffuse)')
    plt.plot(x_nm, phi_gc_theory, 'r--', lw=2, label='Gouy-Chapman (Sharp Interface)')
    plt.title("Electrostatic Potential Comparison")
    plt.xlabel("Distance (nm)")
    plt.ylabel("Potential (V)")
    plt.xlim(center_nm, center_nm + 20) 
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/phi_comparison.png")
    plt.close()

    core_idx = Nx // 2
    phi_0 = phi_phys[core_idx]
    c_vo_c = C_vo_phys[core_idx]
    c_dop_c = C_dop_phys[core_idx]

    print(f"Finished dg_dop = {delta_g_dop_eV:.2f} eV | Phi_0: {phi_0:.4f} V, Vo: {c_vo_c:.2e}, Dop: {c_dop_c:.2e}")

    return phi_0, c_vo_c, c_dop_c

if __name__ == "__main__":
    dg_dop_array = [-1.5]#np.linspace(-1.5, 0.5, 21)
    N_vo_c_list = [1.0e27]
    N_dop_c_list = [1.68e27]

    print(f"Starting parameter sweep. Total runs: {len(dg_dop_array)*len(N_vo_c_list)*len(N_dop_c_list)}")

    case_idx = 1
    for N_vo_c in N_vo_c_list:
        for N_dop_c in N_dop_c_list:
            print(f"\n--- Running Case {case_idx}/6: N_vo_c={N_vo_c:.2e}, N_dop_c={N_dop_c:.2e} ---")
            for dg_dop in dg_dop_array:
                print(f"        with dg_dop={dg_dop:.1f} eV:\n")
                output_dir = f"bicrystal_1d_N_vo_c={N_vo_c:.2e}_N_dop_c={N_dop_c:.2e}_dg_dop={dg_dop:.1f}_eV" 
                if os.path.exists(output_dir): 
                    shutil.rmtree(output_dir) 
                os.makedirs(output_dir)
                phi_0, c_vo_c, c_dop_c = run_bicrystal_1d(
                    output_dir=output_dir, delta_g_dop_eV=dg_dop, N_vo_c_phys=N_vo_c, N_dop_c_phys=N_dop_c
                )
            case_idx += 1