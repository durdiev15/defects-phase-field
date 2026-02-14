import numpy as np
import matplotlib.pyplot as plt

def plot_grains(output_dir, n, eta, Lx, Ly, length_scale, x_nm, p0, p1, E_dens_phys, sig_phys_val):
    # --- PLOTTING ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    Nx, Ny = eta[:,:,0] .shape

    # 1. Grain Structure
    rgb = np.zeros((Nx, Ny, 3))
    rgb[:,:,0] = eta[:,:,0] 
    rgb[:,:,2] = eta[:,:,1]
    ax[0].imshow(np.transpose(rgb, (1,0,2)), extent=[0, Lx*length_scale*1e9, 0, Ly*length_scale*1e9])
    ax[0].set_title(f"Grains (Step {n})")
    ax[0].set_xlabel("x (nm)")
    ax[0].set_ylabel("y (nm)")
    
    # 2. Profiles (Order Parameters)
    ax[1].plot(x_nm, p0, 'r-', label='Grain 1')
    ax[1].plot(x_nm, p1, 'b--', label='Grain 2')
    ax[1].set_title("Order Parameters")
    ax[1].set_xlabel("Distance (nm)")
    ax[1].set_ylabel("$\eta$")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    # 3. Energy Density (Physical)
    ax[2].plot(x_nm, E_dens_phys, 'k-', lw=2)
    ax[2].fill_between(x_nm, E_dens_phys, color='orange', alpha=0.3)
    ax[2].set_title(fr"GB Energy Density\nIntegral $\approx$ {sig_phys_val:.3f} $J/m^2$")
    ax[2].set_xlabel("Distance (nm)")
    ax[2].set_ylabel("Energy Density ($J/m^3$)")
    ax[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/phys_step_{n:05d}.png")
    plt.close(fig)


def plot_fields(output_dir, n, phi, C_vo, C_dop, Lx, Ly, length_scale, phi_scale, conc_scale):
    """
    Plots 2D fields (Top Row) and 1D profiles (Bottom Row).
    Layout:
      [ Phi 2D ] [ C_vo 2D ] [ C_dop 2D ]  (Horizontal Colorbars)
      [ Phi 1D ] [ C_vo 1D ] [ C_dop 1D ]
    """
    
    # 1. Geometry & Scaling
    Nx, Ny = phi.shape
    mid_y = Ny // 2  # Index for 1D profile slice
    
    # Convert Dimensions to nanometers
    Lx_nm = Lx * length_scale * 1e9
    Ly_nm = Ly * length_scale * 1e9
    x_nm = np.linspace(0, Lx_nm, Nx)
    
    # Convert Fields to Physical Units
    phi_phys = phi * phi_scale         # Dimensionless -> Volts
    C_vo_phys = C_vo * conc_scale      # Site fraction -> 1/m^3
    C_dop_phys = C_dop * conc_scale    # Site fraction -> 1/m^3

    # 2. Setup Plot (2 Rows, 3 Columns)
    # height_ratios=[1, 1.5] gives more room to the bottom row plots
    fig, ax = plt.subplots(2, 3, figsize=(18, 10), height_ratios=[1, 1.5])
    extent = [0, Lx_nm, 0, Ly_nm]
    
    # --- COLUMN 1: Electrostatic Potential (Phi) ---
    # Top: 2D Map (Equal Aspect, Horizontal Colorbar)
    # Removing aspect='auto' forces physical aspect ratio (thin strip)
    im0 = ax[0,0].imshow(phi_phys.T, origin='lower', extent=extent, cmap='inferno')
    
    # Create horizontal colorbar
    cbar0 = fig.colorbar(im0, ax=ax[0,0], orientation='horizontal', fraction=0.046, pad=0.2)
    cbar0.set_label('Potential (V)')
    
    ax[0,0].set_title(f"Potential $\phi$ (Step {n})")
    ax[0,0].set_ylabel("y (nm)")
    ax[0,0].set_xlabel("x (nm)")

    # Bottom: 1D Profile
    ax[1,0].plot(x_nm, phi_phys[:, mid_y], 'k-', lw=2)
    ax[1,0].set_title("Potential Profile ($y=L_y/2$)")
    ax[1,0].set_ylabel("$\phi$ (V)")
    ax[1,0].set_xlabel("Distance (nm)")
    ax[1,0].grid(True, alpha=0.3)

    # --- COLUMN 2: Oxygen Vacancy ---
    # Top: 2D Map
    im1 = ax[0,1].imshow(C_vo_phys.T, origin='lower', extent=extent, cmap='viridis')
    cbar1 = fig.colorbar(im1, ax=ax[0,1], orientation='horizontal', fraction=0.046, pad=0.2)
    cbar1.set_label('$C_{V_O}$ ($m^{-3}$)')
    
    ax[0,1].set_title("Oxygen Vacancy $C_{V_O}$")
    ax[0,1].set_ylabel("y (nm)")
    ax[0,1].set_xlabel("x (nm)")

    # Bottom: 1D Profile (Log Scale)
    ax[1,1].plot(x_nm, C_vo_phys[:, mid_y], 'g-', lw=2)
    ax[1,1].set_yscale('log')
    ax[1,1].set_title("$C_{V_O}$ Profile (Log Scale)")
    ax[1,1].set_ylabel("$C_{V_O}$ ($m^{-3}$)")
    ax[1,1].set_xlabel("Distance (nm)")
    ax[1,1].grid(True, alpha=0.3, which="both")

    # --- COLUMN 3: Dopant ---
    # Top: 2D Map
    im2 = ax[0,2].imshow(C_dop_phys.T, origin='lower', extent=extent, cmap='plasma')
    cbar2 = fig.colorbar(im2, ax=ax[0,2], orientation='horizontal', fraction=0.046, pad=0.2)
    cbar2.set_label('$C_{dop}$ ($m^{-3}$)')
    
    ax[0,2].set_title("Dopant $C_{dop}$")
    ax[0,2].set_ylabel("y (nm)")
    ax[0,2].set_xlabel("x (nm)")

    # Bottom: 1D Profile (Log Scale)
    ax[1,2].plot(x_nm, C_dop_phys[:, mid_y], 'b-', lw=2)
    ax[1,2].set_yscale('log')
    ax[1,2].set_title("$C_{dop}$ Profile (Log Scale)")
    ax[1,2].set_ylabel("$C_{dop}$ ($m^{-3}$)")
    ax[1,2].set_xlabel("Distance (nm)")
    ax[1,2].grid(True, alpha=0.3, which="both")

    # Finalize
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fields_step_{n:05d}.png")
    plt.close(fig)


def plot_fields_with_rho(output_dir, n, phi, rho, C_vo, C_dop, Lx, Ly, length_scale, phi_scale, conc_scale):
    """
    Plots 4 columns: Potential, Charge Density, Vo, Dopant
    """
    # Constants
    e_charge = 1.6022e-19
    
    # 1. Geometry & Scaling
    Nx, Ny = phi.shape
    mid_y = Ny // 2
    
    # Convert Dimensions
    Lx_nm = Lx * length_scale * 1e9
    Ly_nm = Ly * length_scale * 1e9
    x_nm = np.linspace(0, Lx_nm, Nx)
    extent = [0, Lx_nm, 0, Ly_nm]
    
    # Convert Fields
    phi_phys = phi * phi_scale
    C_vo_phys = C_vo * conc_scale
    C_dop_phys = C_dop * conc_scale
    
    # Calculate Rho Physical [C/m^3] if passed as dimensionless
    # Assuming 'rho' passed in is already calculated as (z_vo*C_vo + z_dop*C_dop)
    rho_phys = rho * conc_scale * e_charge 

    # 2. Plot Setup (2 Rows, 4 Columns)
    fig, ax = plt.subplots(2, 4, figsize=(24, 10))
    
    # --- COL 1: POTENTIAL ---
    ax[0,0].imshow(phi_phys.T, origin='lower', extent=extent, cmap='inferno', aspect='auto')
    ax[0,0].set_title(f"Potential $\phi$ (V)")
    ax[0,0].set_ylabel("y (nm)")
    
    ax[1,0].plot(x_nm, phi_phys[:, mid_y], 'k-', lw=2)
    ax[1,0].set_title("$\phi$ Profile")
    ax[1,0].set_ylabel("Volts")
    ax[1,0].grid(True, alpha=0.3)

    # --- COL 2: CHARGE DENSITY (RHO) ---
    # Use a diverging colormap (red=pos, blue=neg) centered at 0
    vmax = np.max(np.abs(rho_phys))
    im_rho = ax[0,1].imshow(rho_phys.T, origin='lower', extent=extent, cmap='bwr', aspect='auto', vmin=-vmax, vmax=vmax)
    plt.colorbar(im_rho, ax=ax[0,1], label='$C/m^3$')
    ax[0,1].set_title(r"Charge Density $\rho$")
    
    ax[1,1].plot(x_nm, rho_phys[:, mid_y], 'r-', lw=1.5)
    ax[1,1].axhline(0, color='k', linestyle='--', linewidth=0.5) # Zero line
    ax[1,1].set_title(r"$\rho$ Profile")
    ax[1,1].set_ylabel("$C/m^3$")
    ax[1,1].grid(True, alpha=0.3)

    # --- COL 3: OXYGEN VACANCY ---
    ax[0,2].imshow(C_vo_phys.T, origin='lower', extent=extent, cmap='viridis', aspect='auto')
    ax[0,2].set_title(r"$C_{V_O}$")
    
    ax[1,2].plot(x_nm, C_vo_phys[:, mid_y], 'g-', lw=2)
    ax[1,2].set_yscale('log')
    ax[1,2].set_title(r"$C_{V_O}$ Profile")

    # --- COL 4: DOPANT ---
    ax[0,3].imshow(C_dop_phys.T, origin='lower', extent=extent, cmap='plasma', aspect='auto')
    ax[0,3].set_title(r"$C_{dop}$")
    
    ax[1,3].plot(x_nm, C_dop_phys[:, mid_y], 'b-', lw=2)
    ax[1,3].set_yscale('log')
    ax[1,3].set_title(r"$C_{dop}$ Profile")

    # Formatting x-labels
    for i in range(4):
        ax[0,i].set_xlabel("x (nm)")
        ax[1,i].set_xlabel("Distance (nm)")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fields_rho_step_{n:05d}.png")
    plt.close(fig)