import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# --- Physical Constants ---
T = 1623.0 
R, F, e_charge = 8.314, 96485.0, 1.6022e-19
Vt = (R * T) / F
eps0, eps_r = 8.854e-12, 56.0
eps_total = eps0 * eps_r

z_vo, z_dop = 2.0, -1.0
dg_vo_eV = -1.5

N_vo_b_phys = 5.1e28
N_dop_b_phys = 1.68e28
c_dop_bulk_phys = 7.0e25
c_vo_bulk_phys = -(z_dop * c_dop_bulk_phys) / z_vo

w_c = 0.78e-9 # Core width

mu_vo_bulk = np.log(c_vo_bulk_phys / (N_vo_b_phys - c_vo_bulk_phys))
mu_dop_bulk = np.log(c_dop_bulk_phys / (N_dop_b_phys - c_dop_bulk_phys))
K_vo = np.exp(-dg_vo_eV / Vt)

# --- BVP Solver for Phase-Field Equilibrium ---
def solve_pf_eq_bvp(dg_dop_eV, N_vo_c_phys, N_dop_c_phys, guess_phi_0):
    K_dop = np.exp(-dg_dop_eV / Vt)
    
    def h_func(x_nm):
        x_m = x_nm * 1e-9
        # Core is centered exactly at x=0
        eta0 = 0.5 * (1.0 - np.tanh(x_m / (w_c / 2.0)))
        eta1 = 0.5 * (1.0 + np.tanh(x_m / (w_c / 2.0)))
        sum_eta3 = eta0**3 + eta1**3
        sum_eta2 = eta0**2 + eta1**2
        return (4.0 / 3.0) * (1.0 - 4.0 * sum_eta3 + 3.0 * (sum_eta2**2))

    def ode_sys(x_nm, y):
        phi = y[0]
        dphi_dx_nm = y[1]
        h_val = h_func(x_nm)
        
        # Fermi-Dirac Bulk Concentrations
        C_vo_b = N_vo_b_phys / (1.0 + np.exp(z_vo * phi / Vt - mu_vo_bulk))
        C_dop_b = N_dop_b_phys / (1.0 + np.exp(z_dop * phi / Vt - mu_dop_bulk))
        
        # KKS Core Concentrations
        C_vo_c = (N_vo_c_phys * K_vo * C_vo_b) / (N_vo_b_phys + C_vo_b * (K_vo - 1.0))
        C_dop_c = (N_dop_c_phys * K_dop * C_dop_b) / (N_dop_b_phys + C_dop_b * (K_dop - 1.0))
        
        # Diffuse Interface Interpolation
        C_vo = (1.0 - h_val) * C_vo_b + h_val * C_vo_c
        C_dop = (1.0 - h_val) * C_dop_b + h_val * C_dop_c
        
        rho = e_charge * (z_vo * C_vo + z_dop * C_dop)
        
        # Equation: d2phi_dx2 = -rho / eps 
        d2phi_dx2_nm = -rho / eps_total * (1e-9)**2
        return np.vstack((dphi_dx_nm, d2phi_dx2_nm))

    def bc(ya, yb):
        # Boundary Conditions:
        # ya at x=0 (Symmetry line at Core center): dphi/dx = 0
        # yb at x=L (Far Bulk): phi = 0
        return np.array([ya[1], yb[0]]) 

    # Solves half of the domain: from core (x=0) to bulk (x=100nm)
    x_mesh = np.linspace(0, 100.0, 500) 
    y_guess = np.zeros((2, x_mesh.size))
    
    L_guess = 2.0 
    y_guess[0] = guess_phi_0 * np.exp(-x_mesh / L_guess)
    y_guess[1] = -(guess_phi_0 / L_guess) * np.exp(-x_mesh / L_guess)
    
    res = solve_bvp(ode_sys, bc, x_mesh, y_guess, tol=1e-5, max_nodes=50000)
    if res.success:
        return res.sol(0)[0]
    else:
        return np.nan

# --- Execute Parameter Sweep ---
if __name__ == "__main__":
    dg_dop_array = np.linspace(-1.5, 0.5, 41)
    N_vo_c_list = [1.0e27, 5.0e26]
    N_dop_c_list = [1.68e26, 8.4e26, 1.68e27]
    
    colors = {1.68e26: 'r', 8.4e26: 'b', 1.68e27: 'darkorange'}
    linestyles = {5.0e26: '--', 1.0e27: '-'}
    markers = {5.0e26: '^', 1.0e27: 'x'}
    
    plt.figure(figsize=(7, 6))
    
    print("Sweeping Parameters...")
    for N_vo_c in N_vo_c_list:
        for N_dop_c in N_dop_c_list:
            print(f"Solving N_vo={N_vo_c:.1e}, N_dop={N_dop_c:.1e}")
            phi_0_results = []
            guess = 0.2 
            
            for dg in dg_dop_array:
                phi_0 = solve_pf_eq_bvp(dg, N_vo_c, N_dop_c, guess)
                phi_0_results.append(phi_0)
                if not np.isnan(phi_0):
                    guess = phi_0 # Use current answer as guess for next point to speed up solver
                    
            plt.plot(dg_dop_array, phi_0_results, 
                     color=colors[N_dop_c], 
                     linestyle=linestyles[N_vo_c], 
                     marker=markers[N_vo_c], markevery=4, markersize=7,
                     label=f"N$_{{dop,c}}$={N_dop_c:.1e}, N$_{{V_O,c}}$={N_vo_c:.1e}")
    
    plt.axhline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
    plt.axvline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
    plt.title("GB Potential vs Dopant Segregation Energy")
    plt.xlabel("$\Delta g_{dop}$ (eV)", fontsize=12)
    plt.ylabel("GB Potential $\phi_0$ (V)", fontsize=12)
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure3_reproduction.png", dpi=300)
    print("Saved 'figure3_reproduction.png'")