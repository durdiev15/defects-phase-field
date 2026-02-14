import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def solve_poisson_2d(Nx, Ny, dx_dimless, dy_dimless, C_Vo, C_dop, chi, z_Vo, z_dop):
    """
    Solves 2D Poisson Eq: (d2/dx2 + d2/dy2) phi = -chi * rho
    
    Boundary Conditions (Based on paper benchmark):
    - X Direction: Left Grounded (phi=0), Right Neumann (d_phi/dx=0)
    - Y Direction: Periodic
    """
    
    # Dimensionless charge density field (Nx, Ny)
    rho_dimless = z_Vo * C_Vo + z_dop * C_dop 
    b = -chi * rho_dimless.flatten()
    
    # 1D finite difference stencils
    # x-direction 
    main_x = -2.0 * np.ones(Nx)
    off_x  = np.ones(Nx - 1)
    Dx2 = (np.diag(main_x) + np.diag(off_x, -1) + np.diag(off_x, 1)) / (dx_dimless**2)
    
    # y-direction (Periodic stencil)
    main_y = -2.0 * np.ones(Ny)
    off_y  = np.ones(Ny - 1)
    Dy2 = (np.diag(main_y) + np.diag(off_y, -1) + np.diag(off_y, 1))
    Dy2[0, -1] = 1.0 
    Dy2[-1, 0] = 1.0 
    Dy2 = Dy2 / (dy_dimless**2)

    # Kronecker sum for 2D Laplacian Matrix
    Ix = sp.eye(Nx)
    Iy = sp.eye(Ny)
    
    # --- FIX IS HERE ---
    # We must match C-ordering (Row-Major):
    # - Outer index (Slow) is X -> associated with Dx2
    # - Inner index (Fast) is Y -> associated with Dy2
    L = sp.kron(Dx2, Iy) + sp.kron(Ix, Dy2)
    
    L = L.tolil() 

    # Apply X-Boundary Conditions on the flat index map
    # Note: Your BC logic relies on i * Ny + j, which is C-order.
    # Swapping the kron above makes the Matrix consistent with this loop.
    for j in range(Ny):
        # 1. Left Boundary (i=0): Grounded phi = 0
        row_left = 0 * Ny + j  
        L[row_left, :] = 0.0
        L[row_left, row_left] = 1.0
        b[row_left] = 0.0
        
        # 2. Right Boundary (i=Nx-1): Neumann d_phi/dx = 0
        # This implements (phi[i] - phi[i-1])/dx = 0
        row_right = (Nx - 1) * Ny + j 
        L[row_right, :] = 0.0
        L[row_right, row_right] = 1.0 / dx_dimless
        # (row_right - Ny) correctly points to (i-1, j) in C-order
        L[row_right, row_right - Ny] = -1.0 / dx_dimless
        b[row_right] = 0.0

    # Solve Sparse System
    L = L.tocsc()
    phi_flat = spla.spsolve(L, b)
    
    return phi_flat.reshape(Nx, Ny)

# # --- Physical Parameter Setup ---
# T = 1623.0                    # Temperature (K)
# F = 96485.3                   # Faraday constant (C/mol)
# R = 8.314                     # Gas constant (J/mol*K)
# kb = 1.3806e-23               # Boltzmann constant (J/K)
# e_charge = 1.6022e-19         # Elementary charge (C)
# eps0 = 8.854e-12              # Vacuum permittivity (F/m)
# eps_r = 56.0                  # Relative permittivity at 1623K
# a = 0.39e-9                   # Lattice constant (m)
# V_m = 6.022e23 * (a**3)       # Molar volume

# # Bulk concentration (Example from Table 3: 7x10^25 m^-3)
# c_dop_inf = 7.0e25 

# # --- 1. Calculate Debye Length (Physical) ---
# # Formula from Eq. (14) / Section 2.1 in the paper
# l_D = np.sqrt((eps0 * eps_r * kb * T) / (2 * (e_charge**2) * c_dop_inf))

# # --- 2. Calculate Dimensionless Chi ---
# # This is the coefficient for the non-dimensional Poisson equation
# chi = (a**2 * F**2) / (V_m * eps0 * eps_r * R * T)

# print(f"--- Physical Constants ---")
# print(f"Debye Length: {l_D*1e9:.4f} nm")
# print(f"Dimensionless Chi: {chi:.4e}")