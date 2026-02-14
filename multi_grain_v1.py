import numpy as np
import matplotlib.pyplot as plt
import os

from solver.utils import laplacian_neumann_multi, initial_bicrystal_grain, initial_defect_fields
from solver.energy import compute_properties, compute_h_and_derivative, driving_force_concentrations
from solver.concentrations import concentration_bulk_core, compute_chemical_potential, update_concentration_neumann_bc, compute_kks_mobility, update_concentration_hybrid_bc
from solver.solve_phi import solve_poisson_2d
from solver.plots import plot_grains, plot_fields, plot_fields_with_rho

import shutil 
output_dir = "multi_order_bicrystal" 
if os.path.exists(output_dir): 
    shutil.rmtree(output_dir) 
os.makedirs(output_dir)

def debye_length(eps_r, T, c_dop_inf, kb = 1.3806e-23, eps0=8.854e-12, e_charge=1.6022e-19):
    l_D = np.sqrt((eps0 * eps_r * kb * T) / (2 * (e_charge**2) * c_dop_inf))
    return l_D

def run_bicrystal_multiorder():
    # --- 1. Simulation Setup ---
    # Paper uses 1D domain [-390, 390] nm. We simulate a smaller slice for speed, 
    # but strictly sufficiently large to hold the Space Charge Layer (approx 10-20nm).
    Lx = 80e-9 
    Ly = 2e-9 # 2D strip, Y dimension doesn't matter much for 1D benchmark
    
    # -------------------------------- MATERIALS PARAMETERS ----------------------------------------
    # BENCHMARK SELECTION: 
    # Case 1: MS Model (T=600K, Mobile Vo, Static Dopants)
    # Case 2: GC Model (T=1623K, Mobile Vo, Mobile Dopants)
    MODEL_TYPE = "GC" 
    
    if MODEL_TYPE == "GC":
        T = 1623.0 
        D_dop_switch = 1.0 # Dopants move
    else: # MS
        T = 600.0
        D_dop_switch = 0.0 # Dopants frozen

    R = 8.314
    a = 3.90e-10
    Na = 6.02214076e23
    V_m = Na * (a**3)

    w_c = 0.78e-9
    sigma = 1.0
    
    # Phase Field parameters
    kappa_phys = (3.0 / 4.0) * sigma * w_c
    omega_phys = 6.0 * sigma / w_c

    # Diffusivities
    D_vo_phys = 1.35e-9  
    D_dop_phys = 1.3e-11 

    # Electrostatics
    eps0 = 8.854e-12
    eps_r = 56.0 # Note: eps_r might be temp dependent in paper, check eq.
    F = 96485.0
    
    # Defect Energies (eV)
    delta_g_vo_eV = -1.5 
    delta_g_dop_eV = -0.5 # Non-zero for GC test
    
    # Charge
    z_vo = 2.0
    z_dop = -1.0
    
    # Site Densities (1/m^3)
    N_vo_b_phys = 5.1e28
    N_dop_b_phys = 1.68e28
    N_vo_c_phys = 5.0e26
    N_dop_c_phys = 1.68e27
    
    # Bulk Concentrations (1/m^3)
    c_dop_bulk_phys = 7.0e25
    c_vo_bulk_phys = -(z_dop * c_dop_bulk_phys) / z_vo 

    # ------------------- Scaling ------------------------
    length_scale = a
    energy_scale = (R * T) / V_m    
    time_scale   = (a**2) / D_vo_phys
    phi_scale    = (R * T) / F

    # Dimensionless Parameters
    Lx_dl = Lx / length_scale
    Ly_dl = Ly / length_scale
    
    kappa = kappa_phys / (length_scale**2 * energy_scale)
    omega = omega_phys / energy_scale
    
    # Chi for Poisson: grad^2 phi = -chi * rho
    # Paper Eq 40/41 -> Constants factor out
    chi_poisson = (length_scale**2 * F * F) / (V_m * eps0 * eps_r * R * T)

    # Normalized Energies
    delta_g_vo = (delta_g_vo_eV * F) / (R * T)
    delta_g_dop = (delta_g_dop_eV * F) / (R * T)
    
    # Normalized Densities/Concentrations (Site Fractions)
    # N_dim = N_phys * a^3
    N_vo_b = N_vo_b_phys * a**3
    N_dop_b = N_dop_b_phys * a**3
    N_vo_c = N_vo_c_phys * a**3
    N_dop_c = N_dop_c_phys * a**3
    
    c_dop_init = c_dop_bulk_phys * a**3
    c_vo_init = c_vo_bulk_phys * a**3

    # Normalized Diffusivities
    # D_dim = D_phys * time_scale / length_scale^2
    # Since time_scale = a^2 / D_vo_phys, D_vo_dim = 1.0
    D_vo_dim = 1.0
    D_dop_dim = (D_dop_phys / D_vo_phys) * D_dop_switch

    L_mobility = 1.0 # Interface Mobility
    gamma = 1.5

    # Grid Setup
    Nx = 512 # Sufficient for 1D profile
    Ny = 2   # Minimal for strip
    dx = Lx_dl / Nx
    dy = Ly_dl / Ny
    
    # Stability
    dt_stab = (dx**2) / (4.0 * L_mobility * kappa)
    dt = 1.0 * dt_stab

    print(f"--- Setup ({MODEL_TYPE}) ---")
    print(f"Grid: {Nx}x{Ny}, dx={dx:.2f}, dt={dt:.4e}")
    print(f"Debye Length (approx): {np.sqrt(1/chi_poisson/c_dop_init)*length_scale*1e9:.2f} nm")
    print(f"Dopant Segregation Energy: {delta_g_dop:.2f} kT")

    kb = R / Na
    e_charge = F / Na
    l_D_phys = np.sqrt((eps0 * eps_r * kb * T) / (2 * e_charge**2 * c_dop_bulk_phys))
    print(f"Calculated Debye Length: {l_D_phys*1e9:.4f} nm")

    # Initialize
    eta, x = initial_bicrystal_grain(Lx_dl, Ly_dl, Nx, Ny, num_grains=2)
    C_dop, C_vo, phi = initial_defect_fields(Nx, Ny, c_dop_init, z_dop, z_vo)
    
    # Initial g0 values (standard formation energies in bulk). 
    # In KKS, we often assume g0_bulk = 0 reference, and g0_core = delta_g.
    g0_vo_b, g0_dop_b = 0.0, 0.0

    steps = 5000001
    for n in range(steps):
        if n == 10000:
            L_mobility = 0.0
        
        # 1. Interpolation Functions
        h, dh_deta = compute_h_and_derivative(eta)
        
        # 2. KKS Partitioning (Get Phase Concentrations)
        C_vo_b, C_vo_c = concentration_bulk_core(delta_g_vo, C_vo, N_vo_b, N_vo_c, h)
        C_dop_b, C_dop_c = concentration_bulk_core(delta_g_dop, C_dop, N_dop_b, N_dop_c, h)
        
        # 3. Solve Poisson Equation
        # Recalculate local Density N_site for Poisson RHS if needed, 
        # But Eq 41 uses phase concentrations weighted by h.
        # solve_poisson_2d now expects Site Densities to map C -> Charge correctly?
        # Actually, in KKS C is conserved. The charge density is z*C_total.
        # But site density constraints apply to limits.
        # Simple Poisson: rho = z_vo * C_vo + z_dop * C_dop
        phi = solve_poisson_2d(Nx, Ny, dx, dy, C_vo, C_dop, chi_poisson, z_vo, z_dop)
        
        # 4. Calculate Electrochemical Potentials (Driving Force for Diffusion)
        # mu = g0 + ln(C/(N-C)) + z*phi
        mu_vo = compute_chemical_potential(C_vo_b, N_vo_b, z_vo, phi, g0_vo_b)
        mu_dop = compute_chemical_potential(C_dop_b, N_dop_b, z_dop, phi, g0_dop_b)
        
        # 5. Evolve Concentrations (Cahn-Hilliard) --- MISSING IN YOUR CODE
        # Calculate variable mobility M = D * (1-C/N)C (Ideal soln derivative approx)
        M_vo = compute_kks_mobility(C_vo_b, C_vo_c, h, N_vo_b, N_vo_c, D_vo_dim)
        M_dop = compute_kks_mobility(C_dop_b, C_dop_c, h, N_dop_b, N_dop_c, D_dop_dim)
        
        C_vo = update_concentration_hybrid_bc(C_vo, mu_vo, M_vo, dt, dx)
        C_dop = update_concentration_hybrid_bc(C_dop, mu_dop, M_dop, dt, dx)
        
        # 6. Evolve Microstructure (Allen-Cahn)
        # Calculate Coupling Driving Force (Grand Potential Difference)
        # omega = N * log(1 - C/N)
        omega_diff_vo = driving_force_concentrations(C_vo_b, C_vo_c, N_vo_b, N_vo_c)
        omega_diff_dop = driving_force_concentrations(C_dop_b, C_dop_c, N_dop_b, N_dop_c)
        
        # F_coupling = h'(eta) * (omega_b - omega_c)
        driving_force_scalar = omega_diff_vo + omega_diff_dop
        F_coupling = dh_deta * driving_force_scalar[:, :, np.newaxis]
        
        # Double Well
        sum_sq = np.sum(eta**2, axis=2, keepdims=True)
        df_bulk = omega * (eta**3 - eta)
        df_inter = 2.0 * omega * gamma * eta * (sum_sq - eta**2)
        
        # Total LHS: dF/deta = df_bulk + df_inter + F_coupling
        # Update: eta_new = eta_old - L * dt * (dF/deta - kappa*laplacian)
        lap = laplacian_neumann_multi(eta, dx)
        rhs = -L_mobility * (df_bulk + df_inter + F_coupling - kappa * lap)
        
        eta = eta + dt * rhs
        
        if n % 10000 == 0:
            # print(f"Step {n}, Max Phi: {np.max(phi):.4f}, Max Vo: {np.max(C_vo):.4e}")
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

            phi_phys_max = (phi) * phi_scale
            C_vo_phys_max = (C_vo) / (length_scale**3)

            print(f"Step {n}:")
            print(f"  Max Phi: {np.max(phi_phys_max):.4f} V, Min Phi: {np.min(phi_phys_max):.4f} V, Max Vo: {np.max(C_vo_phys_max):.4e} 1/m3 | Sigma (Calc): {sig_phys_val:.4f} J/m^2 | Width (Calc): {wid_phys_val*1e9:.4f} nm ")

            # plot_grains(output_dir, n, eta, Lx, Ly, length_scale, x_nm, p0, p1, E_dens_phys, sig_phys_val)
            conc_scale_factor = 1.0 / (a**3)
            # plot_fields(output_dir, n, phi, C_vo, C_dop, Lx_dl, Ly_dl, length_scale, phi_scale, conc_scale_factor)
            rho_dim = z_vo * C_vo + z_dop * C_dop
            plot_fields_with_rho(output_dir, n, phi, rho_dim, C_vo, C_dop, Lx_dl, Ly_dl, length_scale, phi_scale, conc_scale_factor)
    
    # Verify GB energy and width
    sigma_sim_phys = np.sqrt(2)/3 * np.sqrt(kappa * omega) * energy_scale * length_scale
    width_sim_phys = np.sqrt(8*kappa/omega) * length_scale

    print(f"\nReal GB properties: sigma={sigma_sim_phys:.4f} J/m^2 | Width={width_sim_phys/1e-9:.4f} nm\n")

    print("Multi-Order Bicrystal Simulation Complete.")


if __name__ == "__main__":
    run_bicrystal_multiorder()