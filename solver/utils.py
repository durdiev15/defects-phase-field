import numpy as np

def laplacian_neumann_multi(field, dx):
    """
    Computes Laplacian for a 3D array (Nx, Ny, N_grains) with Neumann BCs.
    """
    # Pad x and y dimensions with edge values
    fp = np.pad(field, ((1,1), (1,1), (0,0)), mode='edge')
    
    left  = fp[0:-2, 1:-1, :]
    right = fp[2:,   1:-1, :]
    down  = fp[1:-1, 0:-2, :]
    up    = fp[1:-1, 2:,   :]
    center = fp[1:-1, 1:-1, :]
    
    return (left + right + down + up - 4.0 * center) / (dx**2)

def initial_bicrystal_grain(Lx, Ly, Nx, Ny, num_grains=2):
    # --- 3. Initial Condition: Vertical Interface ---
    eta = np.zeros((Nx, Ny, num_grains))
    
    x = np.linspace(0, Lx, Nx)
    center = Lx / 2.0
    width_init = 5.0
    
    # Create smooth crossing profiles
    # Grain 0: 1 on Left -> 0 on Right
    # Grain 1: 0 on Left -> 1 on Right
    
    # We use a helper X grid that broadcasts to (Nx, Ny)
    X_grid = np.meshgrid(x, np.linspace(0,1,Ny), indexing='ij')[0]
    
    eta[:,:,0] = 0.5 * (1.0 - np.tanh((X_grid - center)/(width_init/2.0)))
    eta[:,:,1] = 0.5 * (1.0 + np.tanh((X_grid - center)/(width_init/2.0)))

    return eta, x

def initial_defect_fields(Nx, Ny, c_dop_bulk, z_dop, z_vo):
    C_dop = np.ones((Nx, Ny)) * c_dop_bulk
    ratio = - z_dop / z_vo 
    C_vo = C_dop * ratio
    phi = np.zeros((Nx, Ny))
    return C_dop, C_vo, phi

def debye_length(eps_r, T, c_dop_inf, kb = 1.3806e-23, eps0=8.854e-12, e_charge=1.6022e-19):
    l_D = np.sqrt((eps0 * eps_r * kb * T) / (2 * (e_charge**2) * c_dop_inf))
    print(f"Debye Length: {l_D*1e9:.4f} nm")
    return l_D