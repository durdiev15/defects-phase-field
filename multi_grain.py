import numpy as np
import matplotlib.pyplot as plt
import os

from solver.plots import plot_grains
from solver.utils import laplacian_neumann_multi, initial_bicrystal_grain, initial_defect_fields
from solver.energy import compute_properties, compute_h_and_derivative, driving_force_concentrations
from solver.concentrations import concentration_bulk_core
from solver.solve_phi import solve_poisson_2d

output_dir = "multi_order_bicrystal"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def run_bicrystal_multiorder():
    # --- 1. Simulation Setup ---
    Lx = 40e-9 # in m
    Ly = 10e-9  # in m
    
    # -------------------------------- MATERIALS PARAMETERS ----------------------------------------
    T = 1623.0;          # K
    R = 8.314;           # J/(mol K)
    a = 3.90e-10;        # m (Lattice constant)
    Na = 6.02214076e23   # 1/mol
    V_m = Na * (a * a * a)

    w_c = 0.78e-9;       # m (GB width)
    sigma = 1.0;         # J/m^2 (GB Energy)
    kappa_phys = (3.0 / 4.0) * sigma * w_c
    omega_phys = 6.0 * sigma / w_c

    D_vo_phys = 1.35e-9   # m^2/s
    D_dop_phys = 1.3e-11  # m^2/s

    eps0 = 8.854e-12  # vacuum permittivity F/m
    eps_r = 56.0  # Relative Permittivity
    F = 96485.0   # C/mol
    delta_g_vo_eV = -1.5 # eV seggregation energy
    delta_g_dop_eV = -0.0 # eV seggregation energy
    z_vo = 2.0
    z_dop = -1.0
    N_vo_b_phys = 5.1e28   # Site Densities (Physical 1/m^3)
    N_dop_b_phys = 1.68e28
    N_vo_c_phys = 5.0e26
    N_dop_c_phys = 1.68e27
    c_dop_bulk_phys = 7.0e25   # Bulk Concentrations (Physical 1/m^3)
    c_vo_bulk_phys = -(z_dop * c_dop_bulk_phys) / z_vo # Electroneutrality condition: 2[VO] - 1[Dop] = 0

    # ------------------- Scaling parameters ------------------------
    length_scale = a
    energy_scale = (R * T) / V_m    
    time_scale   = (a * a) / D_vo_phys
    phi_scale    = (R * T) / F

    # -------------------- Dimensionless ----------------------------
    Lx = Lx / length_scale
    Ly = Ly / length_scale

    kappa = kappa_phys / (length_scale * length_scale * energy_scale)
    omega = omega_phys / energy_scale
    K_poisson = (length_scale * length_scale * F * F) / (V_m * eps0 * eps_r * R * T)  # K = (L^2 * F^2) / (Vm * eps0 * eps_r * R * T)
    delta_g_vo = (delta_g_vo_eV * F) / (R * T)
    delta_g_dop = (delta_g_dop_eV * F) / (R * T)
    N_vo_b = N_vo_b_phys * a**3
    N_dop_b = N_dop_b_phys * a**3
    N_vo_c = N_vo_c_phys * a**3
    N_dop_c = N_dop_c_phys * a**3
    c_dop_bulk = c_dop_bulk_phys * a**3
    c_vo_bulk = c_vo_bulk_phys * a**3

    L = 1.0         # Mobility dimensionless
    gamma = 1.5     # Interaction 

    Nx = 1024
    Ny = 16      # Thin strip
    dx = Lx / Nx
    dy = Ly / dy
        
    # Time Stepping
    dt_stability_limit = (dx**2) / (4.0 * L * kappa)
    dt = 0.2 * dt_stability_limit
    
    print(f"--- Stability Check ---")
    print(f"dx = {dx:.5f}")
    print(f"Max Stable dt = {dt_stability_limit:.6f}")
    print(f"Using dt = {dt:.6f}")

    total_time = 200.0  # Decide how long you want to run physically
    steps = 6000 # int(total_time / dt)
    plot_interval = steps // 10  # Plot 10 frames total
    print(f"--- Multi-Order Bicrystal Setup ---")
    print(f"Grid: {Nx}x{Ny}, dx={dx}, steps={steps}")
    print(f"Params: Omega={omega}, Gamma={gamma}, Kappa={kappa}")

    eta, x = initial_bicrystal_grain(Lx, Ly, Nx, Ny, num_grains=2)
    C_dop, C_vo, phi = initial_defect_fields(Nx, Ny, c_dop_bulk, z_dop, z_vo)

    for n in range(steps):

        # 1. Geometry
        h, dh_deta = compute_h_and_derivative(eta)

        # 2. Thermodynamics (Phase Concentrations)
        C_vo_b, C_vo_c = concentration_bulk_core(delta_g0=delta_g_vo, C=C_vo, Nb=N_vo_b, Nc=N_vo_c, h=h)
        C_dop_b, C_dop_c = concentration_bulk_core(delta_g0=delta_g_dop, C=C_dop, Nb=N_dop_b, Nc=N_dop_c, h=h)
        
        # 3. Electrostatics (Finite Difference Solver)
        phi = solve_poisson_2d(Nx=Nx, Ny=Ny, dx_dimless=dx, dy_dimless=dy, C_Vo=C_vo, C_dop=C_dop, chi=K_poisson)

        # 4. Calculate Driving Forces
        diff_vo = driving_force_concentrations(C_vo_b, C_vo_c, N_vo_b, N_vo_c)
        diff_dop = driving_force_concentrations(C_dop_b, C_dop_c, N_dop_b, N_dop_c)
        F_driving = dh_deta * (diff_vo + diff_dop)

        # A. Laplacian
        lap = laplacian_neumann_multi(eta, dx)
        sum_sq = np.sum(eta**2, axis=2, keepdims=True)
        df_bulk = omega * (eta**3 - eta)
        interaction_sum = sum_sq - eta**2
        df_inter = 2.0 * omega * gamma * eta * interaction_sum
        driving_force_chem = 0.0

        # Total derivative
        df = df_bulk + df_inter + driving_force_chem

        # Allen–Cahn RHS
        rhs = L * (kappa * lap - df)

        eta = eta + dt * rhs
        
        # --- Analysis ---
        if n % plot_interval == 0:
            sig_dim, wid_dim, E_dens_dim, p0, p1 = compute_properties(eta, dx, omega, gamma, kappa)
            
            # --- CONVERT TO PHYSICAL UNITS ---
            # Sigma [J/m^2] = Sigma_dim * Energy_Scale [J/m^3] * Length_Scale [m]
            sig_phys_val = sig_dim * energy_scale * length_scale
            
            # Width [m] = Width_dim * Length_Scale [m]
            wid_phys_val = wid_dim * length_scale
            
            # X-axis [nm]
            x_nm = x * length_scale * 1e9 
            
            # Energy Density [J/m^3]
            E_dens_phys = E_dens_dim * energy_scale

            print(f"Step {n}:")
            print(f"  Sigma (Calc): {sig_phys_val:.4f} J/m^2 | Width (Calc): {wid_phys_val*1e9:.4f} nm")

            plot_grains(output_dir, n, eta, Lx, Ly, length_scale, x_nm, p0, p1, E_dens_phys, sig_phys_val)
    
    # Verify GB energy and width
    sigma_sim_phys = np.sqrt(2)/3 * np.sqrt(kappa * omega) * energy_scale * length_scale
    width_sim_phys = np.sqrt(8*kappa/omega) * length_scale

    print(f"\nSimulated GB properties: sigma={sigma_sim_phys:.4f} J/m^2 | Width={width_sim_phys/1e-9:.4f} nm\n")

    print("Multi-Order Bicrystal Simulation Complete.")


if __name__ == "__main__":
    run_bicrystal_multiorder()