import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, quad
from scipy.optimize import root_scalar

# --- Physical Constants ---
T = 1623.0 
R, F, e_charge = 8.314, 96485.0, 1.6022e-19
Vt = (R * T) / F
eps0, eps_r = 8.854e-12, 56.0
eps_total = eps0 * eps_r

z_vo, z_dop = 2.0, -1.0
dg_vo_eV = -1.5
dg_dop_eV = -1.5

N_vo_c_phys = 1.0e27
N_vo_b_phys = 5.1e28
N_dop_b_phys = 1.68e28
c_dop_bulk_phys = 7.0e25
c_vo_bulk_phys = -(z_dop * c_dop_bulk_phys) / z_vo

w_c = 0.78e-9

mu_vo_bulk = np.log(c_vo_bulk_phys / (N_vo_b_phys - c_vo_bulk_phys))
mu_dop_bulk = np.log(c_dop_bulk_phys / (N_dop_b_phys - c_dop_bulk_phys))
K_vo = np.exp(-dg_vo_eV / Vt)
K_dop = np.exp(-dg_dop_eV / Vt)

# --- Define Nx and Lx ---
# In this specific script (which uses SciPy's solve_bvp), Nx and Lx are implicitly 
# defined by the spatial mesh passed to the solver.
# We can define them explicitly here to control the solver's initial mesh.

Lx_nm = 40.0  # The length of the domain to solve (in nanometers). 10 nm is enough for the SCL.
Nx = 5500     # The number of grid points for the initial mesh.

# --- 1. GC Sharp Interface Exact Root Finder ---
def get_gc_phi0(N_dop_c_phys):
    def c_core(phi, N_c, N_b, mu_b, z, K):
        C_b = N_b / (1.0 + np.exp(z * phi / Vt - mu_b))
        return (N_c * K * C_b) / (N_b + C_b * (K - 1.0))
    
    def c_bulk(phi, N_b, mu_b, z):
        return N_b / (1.0 + np.exp(z * phi / Vt - mu_b))
    
    def rho_bulk(phi):
        return e_charge * (z_vo * c_bulk(phi, N_vo_b_phys, mu_vo_bulk, z_vo) + 
                           z_dop * c_bulk(phi, N_dop_b_phys, mu_dop_bulk, z_dop))
        
    def rho_core(phi):
        return e_charge * (z_vo * c_core(phi, N_vo_c_phys, N_vo_b_phys, mu_vo_bulk, z_vo, K_vo) + 
                           z_dop * c_core(phi, N_dop_c_phys, N_dop_b_phys, mu_dop_bulk, z_dop, K_dop))

    def Q_SC(phi):
        integral, _ = quad(rho_bulk, 0, phi)
        val = -2.0 * integral
        if val < 0: val = 0
        E_surf = np.sqrt(val / eps_total)
        return -np.sign(phi) * eps_total * E_surf

    def charge_balance(phi):
        Sigma_core = rho_core(phi) * w_c
        return Sigma_core + 2 * Q_SC(phi)

    res = root_scalar(charge_balance, bracket=[-1.0, 1.0])
    return res.root

# --- 2. Solve the ODE for the GC spatial profile ---
def solve_gc_profile(phi_max):
    def ode_sys(x_nm, y):
        phi = y[0]
        dphi_dx = y[1]
        C_vo_b = N_vo_b_phys / (1.0 + np.exp(z_vo * phi / Vt - mu_vo_bulk))
        C_dop_b = N_dop_b_phys / (1.0 + np.exp(z_dop * phi / Vt - mu_dop_bulk))
        rho = e_charge * (z_vo * C_vo_b + z_dop * C_dop_b)
        d2phi_dx2 = -rho / eps_total * (1e-9)**2
        return np.vstack((dphi_dx, d2phi_dx2))
    def bc(ya, yb):
        return np.array([ya[0] - phi_max, yb[0]])
    
    # Use the defined Lx and a smaller Nx for GC since it's simpler
    x_mesh = np.linspace(0, Lx_nm, Nx // 3) 
    y_guess = np.zeros((2, x_mesh.size))
    y_guess[0] = phi_max * np.exp(-x_mesh / 2.0)
    y_guess[1] = -(phi_max / 2.0) * np.exp(-x_mesh / 2.0)
    
    res = solve_bvp(ode_sys, bc, x_mesh, y_guess, tol=1e-5, max_nodes=10000)
    return res.x, res.y[0]

# --- 3. Solve the ODE for the Phase-Field spatial profile ---
def solve_pf_profile(N_dop_c_phys, guess_phi_0):
    def h_func(x_nm):
        x_m = x_nm * 1e-9
        eta0 = 0.5 * (1.0 - np.tanh(x_m / (w_c / 1.0)))
        eta1 = 0.5 * (1.0 + np.tanh(x_m / (w_c / 1.0)))
        sum_eta3 = eta0**3 + eta1**3
        sum_eta2 = eta0**2 + eta1**2
        return (4.0 / 3.0) * (1.0 - 4.0 * sum_eta3 + 3.0 * (sum_eta2**2))

    def ode_sys(x_nm, y):
        phi = y[0]
        dphi_dx_nm = y[1]
        h_val = h_func(x_nm)
        
        C_vo_b = N_vo_b_phys / (1.0 + np.exp(z_vo * phi / Vt - mu_vo_bulk))
        C_dop_b = N_dop_b_phys / (1.0 + np.exp(z_dop * phi / Vt - mu_dop_bulk))
        
        C_vo_c = (N_vo_c_phys * K_vo * C_vo_b) / (N_vo_b_phys + C_vo_b * (K_vo - 1.0))
        C_dop_c = (N_dop_c_phys * K_dop * C_dop_b) / (N_dop_b_phys + C_dop_b * (K_dop - 1.0))
        
        C_vo = (1.0 - h_val) * C_vo_b + h_val * C_vo_c
        C_dop = (1.0 - h_val) * C_dop_b + h_val * C_dop_c
        
        rho = e_charge * (z_vo * C_vo + z_dop * C_dop)
        d2phi_dx2_nm = -rho / eps_total * (1e-9)**2
        return np.vstack((dphi_dx_nm, d2phi_dx2_nm))

    def bc(ya, yb):
        return np.array([ya[1], yb[0]]) 

    # Use the explicitly defined Lx and Nx here!
    x_mesh = np.linspace(0, Lx_nm, Nx)
    y_guess = np.zeros((2, x_mesh.size))
    L_guess = 2.0
    y_guess[0] = guess_phi_0 * np.exp(-x_mesh / L_guess)
    y_guess[1] = -(guess_phi_0 / L_guess) * np.exp(-x_mesh / L_guess)
    
    res = solve_bvp(ode_sys, bc, x_mesh, y_guess, tol=1e-5, max_nodes=50000)
    return res.x, res.y[0]

# --- 4. Generate the Plot ---
N_dop_list = [1.68e26, 8.4e26, 1.68e27]
colors = ['r', 'b', 'darkorange']

plt.figure(figsize=(8, 6))

for N_dop, col in zip(N_dop_list, colors):
    # Calculate Phase-Field Profile
    phi0_gc = get_gc_phi0(N_dop)
    x_pf, phi_pf = solve_pf_profile(N_dop, guess_phi_0=phi0_gc)
    plt.plot(x_pf, phi_pf, color=col, linestyle='-', lw=2.5, label=f'PF: $N_{{dop,c}}={N_dop:.1e}$')
    
    # Calculate GC Profile
    x_gc_decay, phi_gc_decay = solve_gc_profile(phi0_gc)
    x_gc_shifted = x_gc_decay + (w_c * 1e9 / 2.0)
    x_gc_core = np.linspace(0, w_c * 1e9 / 2.0, 10)
    phi_gc_core = np.full_like(x_gc_core, phi0_gc)
    
    x_gc_full = np.concatenate([x_gc_core, x_gc_shifted])
    phi_gc_full = np.concatenate([phi_gc_core, phi_gc_decay])
    
    plt.plot(x_gc_full, phi_gc_full, color=col, linestyle='--', lw=2, dashes=(4, 2), label=f'GC: $N_{{dop,c}}={N_dop:.1e}$')

plt.xlim(0, 20)
plt.ylim(0, 0.22)
plt.xlabel('Distance from GB core center (nm)', fontsize=12)
plt.ylabel('Electrostatic Potential $\phi$ (V)', fontsize=12)
plt.title('Spatial Profile of Potential (Reproduction of Fig 3d)', fontsize=13)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure_3d_reproduction.png', dpi=300)
print(f"Figure 3d reproduced successfully with explicit Nx={Nx} and Lx={Lx_nm} nm!")