import numpy as np

def compute_h_and_derivative(eta):
    """
    Computes the interpolation function h(eta) and its derivative dh/deta_i
    based on Eq. (24) in the paper.
    
    eta: 3D array (Nx, Ny, num_grains)
    
    Returns:
    h: 2D array (Nx, Ny) - The scalar interpolation field
    dh_deta: 3D array (Nx, Ny, num_grains) - The derivative field
    """
    # Pre-calculate sums for efficiency
    # sum_eta3 = sum(eta_k^3)
    sum_eta3 = np.sum(eta**3, axis=2, keepdims=True)
    
    # sum_eta2 = sum(eta_k^2)
    sum_eta2 = np.sum(eta**2, axis=2, keepdims=True)
    
    # --- Compute h(eta) ---
    # Eq (24): h = 4/3 * [ 1 - 4*sum(eta^3) + 3*(sum(eta^2))^2 ]
    # We squeeze the last dimension since h is a scalar field
    h = (4.0 / 3.0) * (1.0 - 4.0 * sum_eta3 + 3.0 * (sum_eta2**2))
    h = np.squeeze(h, axis=2) 
    
    # --- Compute derivative dh/deta_i ---
    # Derivative: 16 * eta_i * (sum(eta^2) - eta_i)
    dh_deta = 16.0 * eta * (sum_eta2 - eta)
    
    return h, dh_deta

def compute_properties(eta, dx, omega, gamma, kappa):
    """
    Calculates Interface Energy (Sigma) and Width for the Multi-Order model.
    """
    # Take a horizontal slice through the middle
    mid_y = eta.shape[1] // 2
    
    # eta shape is (Nx, Ny, 2)
    eta0 = eta[:, mid_y, 0]
    eta1 = eta[:, mid_y, 1]
    
    # Bulk term for both grains
    f_bulk_0 = eta0**4/4.0 - eta0**2/2.0
    f_bulk_1 = eta1**4/4.0 - eta1**2/2.0
    
    # Interaction term (eta0^2 * eta1^2)
    f_inter = gamma * (eta0**2) * (eta1**2) + 1/4
    
    f_local = (f_bulk_0 + f_bulk_1 + f_inter) * omega

    # --- 2. Calculate Gradient Energy Density ---
    # f_grad = sum[ 0.5 * kappa * |grad(eta)|^2 ]
    grad_eta0 = np.gradient(eta0, dx)
    grad_eta1 = np.gradient(eta1, dx)
    
    f_grad = 0.5 * kappa * (grad_eta0**2 + grad_eta1**2)
    
    # --- 3. Total Interface Energy (Sigma) ---
    total_energy_density = f_local + f_grad
    sigma_calculated = np.sum(total_energy_density) * dx
    
    # --- 4. Interface Width ---
    # Defined by the slope of the crossing point
    max_slope = np.max(np.abs(grad_eta0))
    width_calculated = 1.0 / max_slope if max_slope > 1e-6 else 0.0
    
    return sigma_calculated, width_calculated, total_energy_density, eta0, eta1

import torch

def driving_force_concentrations(C_b, C_c, N_b, N_c):
    """
    Calculates the dimensionless diriving force from concentrations
    """
    term = N_b * np.log(1 - C_b / N_b) - N_c * np.log(1 - C_c / N_c) 

    return term
