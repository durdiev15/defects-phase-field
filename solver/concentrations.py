import numpy as np

def concentration_bulk_core(delta_g0, C, Nb, Nc, h, epsilon=1e-6):
    """
    Computes phase concentrations Cb and Cc using NumPy.
    Implements KKS constraints: mu_b = mu_c and C = (1-h)Cb + hCc
    """
    # 1. Partition Coefficient k = exp(delta_g0)
    k = np.exp(delta_g0)
    
    # Initialize output arrays with same shape as C
    Cb = np.zeros_like(C, dtype=float)
    Cc = np.zeros_like(C, dtype=float)
    
    # Masks for different regions to avoid division by zero
    mask_bulk = h < epsilon
    mask_core = h > (1.0 - epsilon)
    mask_inter = ~(mask_bulk | mask_core)
    
    # --- Case 1: Bulk Phase (h ~ 0) ---
    if np.any(mask_bulk):
        C_loc = C[mask_bulk]
        Cb[mask_bulk] = C_loc
        # Cc derived from equilibrium: Cc = (C * Nc) / (k*Nb + C*(1-k))
        num = C_loc * Nc
        den = k * Nb + C_loc * (1.0 - k)
        Cc[mask_bulk] = num / (den + 1e-12)

    # --- Case 2: Core Phase (h ~ 1) ---
    if np.any(mask_core):
        C_loc = C[mask_core]
        Cc[mask_core] = C_loc
        # Cb derived from equilibrium: Cb = (k * C * Nb) / (Nc + C*(k-1))
        num = k * C_loc * Nb
        den = Nc + C_loc * (k - 1.0)
        Cb[mask_core] = num / (den + 1e-12)

    # --- Case 3: Interface Region (0 < h < 1) ---
    if np.any(mask_inter):
        h_loc = h[mask_inter]
        C_loc = C[mask_inter]
        
        # Quadratic Coefficients: A*Cc^2 + B*Cc + D = 0
        A = h_loc * (k - 1.0)
        B = h_loc * Nc + k * Nb * (1.0 - h_loc) - C_loc * (k - 1.0)
        D = -C_loc * Nc
        
        # Handle the Linear Case (if k is very close to 1.0, A is 0)
        linear_idx = np.abs(A) < 1e-12
        quad_idx = ~linear_idx
        
        Cc_res = np.empty_like(C_loc)
        
        # Solve Linear part
        if np.any(linear_idx):
            Cc_res[linear_idx] = -D[linear_idx] / (B[linear_idx] + 1e-12)
            
        # Solve Quadratic part
        if np.any(quad_idx):
            Ai = A[quad_idx]
            Bi = B[quad_idx]
            Di = D[quad_idx]
            
            # Discriminant
            disc = Bi**2 - 4 * Ai * Di
            sqrt_disc = np.sqrt(np.maximum(disc, 0)) # Maximize with 0 to avoid NaNs
            
            # Use the positive root
            Cc_res[quad_idx] = (-Bi + sqrt_disc) / (2 * Ai)
            
        # Enforce physical bounds [0, Nc]
        Cc_res = np.clip(Cc_res, 0.0, Nc)
        Cc[mask_inter] = Cc_res
        
        # Calculate Cb using Mass Balance: Cb = (C - h*Cc) / (1-h)
        Cb[mask_inter] = (C_loc - h_loc * Cc_res) / (1.0 - h_loc)

    return Cb, Cc

def compute_chemical_potential(C_b, N_b, z, phi, g0=0.0):
    """
    Computes mu = g0 + ln( C_b / (N_b - C_b) ) + z * phi
    """
    epsilon = 1e-12

    # Clamp safely
    C_safe = np.clip(C_b, epsilon, N_b - epsilon)

    chemical_term = np.log(C_safe / (N_b - C_safe))
    return g0 + chemical_term + z * phi


def update_concentration_hybrid_bc(C, mu, M, dt, dx):
    """
    Conservative finite-difference update with Hybrid BCs:
    - X Direction: Neumann (Zero-Flux) -> Matches 1D Benchmark
    - Y Direction: Periodic -> Matches Infinite Planar Interface assumption
    """
    
    # -------------------------------------------------------------------
    # 1. X-Direction Fluxes (Neumann / Zero-Flux)
    # -------------------------------------------------------------------
    # J_x is defined on the faces between i and i+1. Shape: (Nx-1, Ny)
    dmu_dx = (mu[1:, :] - mu[:-1, :]) / dx
    M_mid_x = 0.5 * (M[1:, :] + M[:-1, :])
    J_x = -M_mid_x * dmu_dx

    # -------------------------------------------------------------------
    # 2. Y-Direction Fluxes (Periodic)
    # -------------------------------------------------------------------
    # We use np.roll to handle the wrap-around index automatically.
    # J_y[i, j] is the flux leaving cell (i,j) into (i, j+1).
    # At j=Ny-1, it flows into j=0.
    
    mu_up = np.roll(mu, shift=-1, axis=1) # Neighbor at j+1
    M_up  = np.roll(M,  shift=-1, axis=1)
    
    dmu_dy = (mu_up - mu) / dx  # Assuming dy = dx
    M_mid_y = 0.5 * (M + M_up)
    
    J_y = -M_mid_y * dmu_dy # Shape (Nx, Ny)

    # -------------------------------------------------------------------
    # 3. Compute Divergence -div(J)
    # -------------------------------------------------------------------
    minus_div_J = np.zeros_like(C)

    # --- X Contribution (Neumann Logic) ---
    # Inflow from left, Outflow to right
    # Boundary i=0: Inflow=0, Outflow=J_x[0]
    minus_div_J[0, :] -= J_x[0, :] / dx
    
    # Internal: Inflow=J_x[i-1], Outflow=J_x[i]
    minus_div_J[1:-1, :] += (J_x[:-1, :] - J_x[1:, :]) / dx
    
    # Boundary i=Nx-1: Inflow=J_x[-1], Outflow=0
    minus_div_J[-1, :] += J_x[-1, :] / dx

    # --- Y Contribution (Periodic Logic) ---
    # Inflow comes from j-1 (roll J_y forward), Outflow goes to j (J_y)
    J_y_in = np.roll(J_y, shift=1, axis=1)
    
    minus_div_J += (J_y_in - J_y) / dx

    return C + dt * minus_div_J

def update_concentration_neumann_bc(C, mu, M, dt, dx):
    """
    Conservative finite-difference update with variable mobility and Neumann (Zero-Flux) BCs.
    
    Flux J = -M * grad(mu)
    dC/dt = -div(J)
    
    Boundaries: Flux J=0 at the domain edges.
    """
    
    # 1. Calculate Fluxes on INTERFACES (Staggered Grid)
    # -------------------------------------------------------------------
    
    # X-Fluxes: Shape (Nx-1, Ny)
    # J_x[i] exists between cell i and i+1
    dmu_dx = (mu[1:, :] - mu[:-1, :]) / dx
    M_mid_x = 0.5 * (M[1:, :] + M[:-1, :]) # Arithmetic mean mobility at face
    J_x = -M_mid_x * dmu_dx

    # Y-Fluxes: Shape (Nx, Ny-1)
    # J_y[j] exists between cell j and j+1
    # NOTE: Since Y is thin strip/periodic in benchmark, usually Y flux is 0 or periodic.
    # Assuming standard Neumann for simplicity here too:
    dmu_dy = (mu[:, 1:] - mu[:, :-1]) / dx # Assuming dy=dx for uniform grid
    M_mid_y = 0.5 * (M[:, 1:] + M[:, :-1])
    J_y = -M_mid_y * dmu_dy

    # 2. Compute Divergence -div(J)
    # -------------------------------------------------------------------
    
    # Initialize flux divergence array
    minus_div_J = np.zeros_like(C)

    # --- X Direction Contribution ---
    # Internal cells: J_in - J_out = J_left - J_right
    # Note: J_x[i] is flux LEAVING i into i+1 (positive right)
    
    # Inflow from left face (index i-1)
    # Outflow to right face (index i)
    
    # For index 0: Inflow is 0 (Neumann), Outflow is J_x[0]
    minus_div_J[0, :] -= J_x[0, :] / dx
    
    # For internal indices 1 to -2:
    minus_div_J[1:-1, :] += (J_x[:-1, :] - J_x[1:, :]) / dx
    
    # For index -1 (Last cell): Inflow is J_x[-1], Outflow is 0 (Neumann)
    minus_div_J[-1, :] += J_x[-1, :] / dx

    # --- Y Direction Contribution ---
    # For index 0: Inflow is 0 (Neumann), Outflow is J_y[0]
    minus_div_J[:, 0] -= J_y[:, 0] / dx
    
    # For internal indices 1 to -2:
    minus_div_J[:, 1:-1] += (J_y[:, :-1] - J_y[:, 1:]) / dx
    
    # For index -1: Inflow is J_y[-1], Outflow is 0 (Neumann)
    minus_div_J[:, -1] += J_y[:, -1] / dx

    # 3. Update
    return C + dt * minus_div_J

def update_concentration_variable_mobility(C, mu, M, dt, dx):
    """
    Conservative finite-difference update with variable mobility (NumPy version).
    Periodic boundary conditions via np.roll.
    """

    # 1. Neighbor access (periodic)
    M_right = np.roll(M, shift=-1, axis=-2)
    M_left  = np.roll(M, shift=1,  axis=-2)
    M_up    = np.roll(M, shift=-1, axis=-1)
    M_down  = np.roll(M, shift=1,  axis=-1)

    mu_right = np.roll(mu, shift=-1, axis=-2)
    mu_left  = np.roll(mu, shift=1,  axis=-2)
    mu_up    = np.roll(mu, shift=-1, axis=-1)
    mu_down  = np.roll(mu, shift=1,  axis=-1)

    # 2. Fluxes (arithmetic mean)
    J_right = 0.5 * (M + M_right) * (mu_right - mu) / dx
    J_left  = 0.5 * (M_left + M)  * (mu - mu_left)  / dx
    J_up    = 0.5 * (M + M_up)    * (mu_up - mu)    / dx
    J_down  = 0.5 * (M_down + M)  * (mu - mu_down)  / dx

    # 3. Divergence
    div_J = (J_right - J_left + J_up - J_down) / dx

    return C + dt * div_J


def compute_kks_mobility(C_b, C_c, h, N_b, N_c, D_diff):
    """
    Computes mobility M(x) based on KKS stiffness mixture rule.
    """
    inv_stiffness_b = (1.0 - (C_b / N_b)) * C_b
    inv_stiffness_c = (1.0 - (C_c / N_c)) * C_c

    inv_stiffness_eff = (1.0 - h) * inv_stiffness_b + h * inv_stiffness_c

    M = D_diff * inv_stiffness_eff
    return M
