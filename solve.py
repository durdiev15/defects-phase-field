import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

def solve_equilibrium_phase_field():
    # --- Physical Parameters ---
    T = 1623.0 
    R, F, e_charge = 8.314, 96485.0, 1.6022e-19
    Vt = (R * T) / F
    eps0, eps_r = 8.854e-12, 56.0
    eps_total = eps0 * eps_r

    z_vo, z_dop = 2.0, -1.0
    dg_vo_eV, dg_dop_eV = -1.5, -1.5

    # Target concentrations & sites
    N_vo_c_phys = 1.0e27
    N_dop_c_phys = 1.68e26 # Your specific cap from the log
    N_vo_b_phys = 5.1e28
    N_dop_b_phys = 1.68e28
    c_dop_bulk_phys = 7.0e25
    c_vo_bulk_phys = -(z_dop * c_dop_bulk_phys) / z_vo

    # --- Setup the 1D Grid ---
    a = 3.90e-10
    Lx = 800e-9 
    Nx = 10240
    length_scale = a
    x_m = np.linspace(0, Lx, Nx)
    dx_m = x_m[1] - x_m[0]
    center_m = Lx / 2.0
    
    # Phase Field width parameter (w_c)
    w_c = 0.78e-9

    # --- 1. Compute Exact Analytical Phase-Field h(x) ---
    eta0 = 0.5 * (1.0 - np.tanh((x_m - center_m)/(w_c/2.0)))
    eta1 = 0.5 * (1.0 + np.tanh((x_m - center_m)/(w_c/2.0)))
    sum_eta3 = eta0**3 + eta1**3
    sum_eta2 = eta0**2 + eta1**2
    h = (4.0 / 3.0) * (1.0 - 4.0 * sum_eta3 + 3.0 * (sum_eta2**2))

    # --- 2. Thermodynamic Equilibrium Relations ---
    mu_vo_bulk = np.log(c_vo_bulk_phys / (N_vo_b_phys - c_vo_bulk_phys))
    mu_dop_bulk = np.log(c_dop_bulk_phys / (N_dop_b_phys - c_dop_bulk_phys))

    K_vo = np.exp(-dg_vo_eV / Vt)
    K_dop = np.exp(-dg_dop_eV / Vt)

    def get_concentrations(phi_array):
        # Fermi-Dirac link between bulk sites and phi
        C_vo_b = N_vo_b_phys / (1.0 + np.exp(z_vo * phi_array / Vt - mu_vo_bulk))
        C_dop_b = N_dop_b_phys / (1.0 + np.exp(z_dop * phi_array / Vt - mu_dop_bulk))
        
        # KKS core constraints
        C_vo_c = (N_vo_c_phys * K_vo * C_vo_b) / (N_vo_b_phys + C_vo_b * (K_vo - 1.0))
        C_dop_c = (N_dop_c_phys * K_dop * C_dop_b) / (N_dop_b_phys + C_dop_b * (K_dop - 1.0))
        
        # Diffuse Interface Interpolation
        C_vo = (1.0 - h) * C_vo_b + h * C_vo_c
        C_dop = (1.0 - h) * C_dop_b + h * C_dop_c
        return C_vo, C_dop

    # --- 3. The Nonlinear Poisson Residual ---
    def poisson_residual(phi_array):
        C_vo, C_dop = get_concentrations(phi_array)
        rho = e_charge * (z_vo * C_vo + z_dop * C_dop)
        
        # Calculate Laplacian using finite differences
        lap_phi = np.zeros_like(phi_array)
        lap_phi[1:-1] = (phi_array[:-2] - 2.0 * phi_array[1:-1] + phi_array[2:]) / (dx_m**2)
        
        # Apply Boundary Conditions
        lap_phi[0] = phi_array[0] - 0.0                      # Dirichlet: phi=0 in far bulk
        lap_phi[-1] = (phi_array[-2] - phi_array[-1]) / dx_m # Neumann: dphi/dx=0 at symmetry line
        
        # Formulate root equation: Laplacian(phi) - (-rho / epsilon) = 0
        rhs = -rho / eps_total
        rhs[0] = 0.0  # Enforce BC
        rhs[-1] = 0.0 # Enforce BC
        
        return lap_phi - rhs

    # --- 4. Solve! ---
    print("Solving Nonlinear Poisson Equation...")
    phi_guess = 0.2 * np.exp(-np.abs(x_m - center_m) / 2.0e-9) # Provide simple initial guess
    
    sol = root(poisson_residual, phi_guess, method='krylov', tol=1e-8)
    
    phi_eq = sol.x
    C_vo_eq, C_dop_eq = get_concentrations(phi_eq)
    
    print("\n--- EXACT PHASE-FIELD EQUILIBRIUM ---")
    print(f"Max Phi:    {np.max(phi_eq):.4f} V")
    print(f"Max C_vo:   {np.max(C_vo_eq):.4e} 1/m3")
    print(f"Max C_dop:  {np.max(C_dop_eq):.4e} 1/m3")
    
    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(x_m * 1e9, phi_eq, 'k-', lw=2, label='Equilibrium $\phi$')
    plt.xlim(center_m * 1e9 - 10, center_m * 1e9 + 10)
    plt.title("Direct Equilibrium Potential")
    plt.xlabel("Distance (nm)")
    plt.ylabel("Potential (V)")
    plt.grid()
    plt.savefig("test.png")

if __name__ == "__main__":
    solve_equilibrium_phase_field()