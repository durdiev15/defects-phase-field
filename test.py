import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# =============================================================================
# 1. PHYSICAL PARAMETERS
# =============================================================================
T = 1623.0
e_charge = 1.6022e-19
kb = 1.3806e-23
eps0 = 8.854e-12
eps_r = 56.0

z_vo = 2.0
z_dop = -1.0

N_vo_b = 5.1e28
N_dop_b = 1.68e28

c_dop_inf = 7.0e25
c_vo_inf = 3.5e25

dg_vo_eV = -1.5
wc_nm = 0.78  # Core width in nm from the paper

Vt = (kb * T) / e_charge  # Thermal voltage (~0.1398 V)

# =============================================================================
# 2. PHASE-FIELD STEADY STATE SOLVER
# =============================================================================
def solve_phase_field_equilibrium(dg_dop_eV, N_vo_c, N_dop_c):
    
    def calc_c(phi, N_phase, N_bulk, c_inf, dg_eV, z):
        """Calculates equilibrium concentration using Fermi-Dirac statistics"""
        term = (N_bulk / c_inf - 1.0) * np.exp((dg_eV + z * phi) / Vt)
        return N_phase / (1.0 + term)

    def ode_system(x_nm, y):
        # y[0] = phi, y[1] = dphi/dx_nm
        phi = y[0]
        dphi_dx_nm = y[1]
        
        # 1. Phase-Field Interpolation Function h(x)
        # Using analytical KKS profile for a flat stationary boundary
        eta1 = 0.5 * (1.0 - np.tanh(2.0 * x_nm / wc_nm))
        eta2 = 0.5 * (1.0 + np.tanh(2.0 * x_nm / wc_nm))
        sum_eta3 = eta1**3 + eta2**3
        sum_eta2 = eta1**2 + eta2**2
        h_x = (4.0 / 3.0) * (1.0 - 4.0 * sum_eta3 + 3.0 * (sum_eta2**2))
        
        # 2. Calculate local phase concentrations
        c_vo_b = calc_c(phi, N_vo_b, N_vo_b, c_vo_inf, 0.0, z_vo)
        c_vo_c = calc_c(phi, N_vo_c, N_vo_b, c_vo_inf, dg_vo_eV, z_vo)
        C_vo_total = (1.0 - h_x) * c_vo_b + h_x * c_vo_c
        
        c_dop_b = calc_c(phi, N_dop_b, N_dop_b, c_dop_inf, 0.0, z_dop)
        c_dop_c = calc_c(phi, N_dop_c, N_dop_b, c_dop_inf, dg_dop_eV, z_dop)
        C_dop_total = (1.0 - h_x) * c_dop_b + h_x * c_dop_c
        
        # 3. Poisson's Equation
        rho = e_charge * (z_vo * C_vo_total + z_dop * C_dop_total)
        d2phi_dx2_m = -rho / (eps0 * eps_r)
        d2phi_dx2_nm = d2phi_dx2_m * 1e-18 # Scale to nm
        
        return np.vstack((dphi_dx_nm, d2phi_dx2_nm))

    def boundary_conditions(ya, yb):
        # Center of GB core (x=0) must be flat (symmetry): dphi/dx = 0
        # Deep in bulk (x=L), potential must be 0: phi = 0
        return np.array([ya[1], yb[0]])

    # Define domain from core (0) to bulk (20 nm)
    x_bvp = np.linspace(0, 20.0, 500)
    
    # Initial guess
    y_guess = np.zeros((2, x_bvp.size))
    y_guess[0] = 0.2 * np.exp(-x_bvp / 1.5)
    y_guess[1] = -(0.2 / 1.5) * np.exp(-x_bvp / 1.5)

    # Solve the Boundary Value Problem
    res = solve_bvp(ode_system, boundary_conditions, x_bvp, y_guess, tol=1e-5)
    
    if res.success:
        # Evaluate exact core values (x = 0)
        phi_0 = res.y[0][0]
        c_vo_core = calc_c(phi_0, N_vo_c, N_vo_b, c_vo_inf, dg_vo_eV, z_vo)
        c_dop_core = calc_c(phi_0, N_dop_c, N_dop_b, c_dop_inf, dg_dop_eV, z_dop)
        return phi_0, c_vo_core, c_dop_core
    else:
        print("BVP Failed to converge!")
        return np.nan, np.nan, np.nan

# =============================================================================
# 3. REPRODUCE FIGURE 3
# =============================================================================
if __name__ == "__main__":
    dg_dop_array = np.linspace(-1.5, 0.5, 21)
    
    # The 6 cases from the paper
    N_vo_c_list = [1.0e27, 5.0e26]
    N_dop_c_list = [1.68e26, 8.4e26, 1.68e27]
    
    # Plotting setup matching the paper
    styles = {1.0e27: '-', 5.0e26: '--'}
    colors = {1.68e26: 'red', 8.4e26: 'teal', 1.68e27: 'orange'}
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    
    print("Solving Phase-Field Equilibrium states...")
    
    for N_vo_c in N_vo_c_list:
        for N_dop_c in N_dop_c_list:
            
            phi_0_vals, c_vo_vals, c_dop_vals = [], [], []
            
            for dg in dg_dop_array:
                p0, c_vo, c_dop = solve_phase_field_equilibrium(dg, N_vo_c, N_dop_c)
                phi_0_vals.append(p0)
                c_vo_vals.append(c_vo)
                c_dop_vals.append(c_dop)
            
            ls = styles[N_vo_c]
            c = colors[N_dop_c]
            lbl = f"Nv,c={N_vo_c:.1e}, Nd,c={N_dop_c:.1e}"
            
            ax[0].plot(dg_dop_array, phi_0_vals, color=c, linestyle=ls, lw=2, label=lbl)
            ax[1].plot(dg_dop_array, c_dop_vals, color=c, linestyle=ls, lw=2)
            ax[2].plot(dg_dop_array, c_vo_vals, color=c, linestyle=ls, lw=2)

    # Format 3a: GB Potential
    ax[0].set_title(r"(a) Grain Boundary Potential $\Phi_0$")
    ax[0].set_ylabel(r'$\Phi_0$ [V]')
    ax[0].set_xlabel(r'$\Delta g_{dop}$ [eV]')
    ax[0].legend(fontsize=8)
    
    # Format 3b: Dopant Core Concentration
    ax[1].set_title(r"(b) Core Dopant Concentration")
    ax[1].set_yscale('log')
    ax[1].set_ylabel(r'$c_{dop,c}$ [m$^{-3}$]')
    ax[1].set_xlabel(r'$\Delta g_{dop}$ [eV]')
    ax[1].set_ylim(1e22, 1e28)
    
    # Format 3c: Oxygen Vacancy Core Concentration
    ax[2].set_title(r"(c) Core Oxygen Vacancy Concentration")
    ax[2].set_yscale('log')
    ax[2].set_ylabel(r'$c_{V_O,c}$ [m$^{-3}$]')
    ax[2].set_xlabel(r'$\Delta g_{dop}$ [eV]')
    ax[2].set_ylim(4e25, 2e27)
    
    for a in ax:
        a.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("Phase_Field_Figure_3.png", dpi=300)
    print("Done! Saved as Phase_Field_Figure_3.png")
    plt.show()