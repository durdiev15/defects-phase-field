import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np

class TransportSolver:
    def __init__(self, Nx, Ny, dx):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.Eye = sp.eye(Nx * Ny, format='csc')
        self.Lap = self._build_laplacian()

    def _build_laplacian(self):
        """Builds a 2D Laplacian with Neumann X and Periodic Y BCs."""
        # X-direction (Neumann)
        main_x = -2.0 * np.ones(self.Nx)
        main_x[0] = -1.0; main_x[-1] = -1.0
        off_x = np.ones(self.Nx - 1)
        Dx2 = sp.diags([off_x, main_x, off_x], [-1, 0, 1]) / (self.dx**2)
        
        # Y-direction (Periodic)
        main_y = -2.0 * np.ones(self.Ny)
        off_y = np.ones(self.Ny - 1)
        Dy2 = sp.diags([off_y, main_y, off_y], [-1, 0, 1]).tolil()
        Dy2[0, -1] = 1.0; Dy2[-1, 0] = 1.0
        Dy2 = (Dy2 / (self.dx**2)).tocsc()

        Ix = sp.eye(self.Nx)
        Iy = sp.eye(self.Ny)
        return (sp.kron(Dx2, Iy) + sp.kron(Ix, Dy2)).tocsc()

    def solve_step(self, C, mu, M, dt, S_stab):
        # Calculate explicit divergence: div(M grad mu)
        dmu_dx = (mu[1:, :] - mu[:-1, :]) / self.dx
        M_mid_x = 0.5 * (M[1:, :] + M[:-1, :])
        J_x = -M_mid_x * dmu_dx

        mu_up = np.roll(mu, shift=-1, axis=1)
        M_up = np.roll(M, shift=-1, axis=1)
        J_y = -0.5 * (M + M_up) * (mu_up - mu) / self.dx

        explicit_div = np.zeros_like(C)
        explicit_div[0, :] -= J_x[0, :] / self.dx
        explicit_div[1:-1, :] += (J_x[:-1, :] - J_x[1:, :]) / self.dx
        explicit_div[-1, :] += J_x[-1, :] / self.dx
        explicit_div += (np.roll(J_y, shift=1, axis=1) - J_y) / self.dx

        # Linear System: (I - dt * S * Lap) C_new = C_old + dt * (div_M_grad_mu - S * Lap * C_old)
        C_flat = C.flatten()
        b = C_flat + dt * (explicit_div.flatten() - S_stab * self.Lap.dot(C_flat))
        A = self.Eye - (dt * S_stab) * self.Lap
        return spla.spsolve(A, b).reshape(self.Nx, self.Ny)
    
class PhaseFieldSolver:
    def __init__(self, Nx, Ny, dx):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.Lap = self._build_laplacian()
        self.Eye = sp.eye(Nx * Ny, format='csc')

    def _build_laplacian(self):
        # Build Laplacian with Neumann X and Periodic Y 
        main_x = -2.0 * np.ones(self.Nx); main_x[0] = -1.0; main_x[-1] = -1.0
        off_x = np.ones(self.Nx - 1)
        Dx2 = sp.diags([off_x, main_x, off_x], [-1, 0, 1]) / (self.dx**2)
        
        main_y = -2.0 * np.ones(self.Ny)
        off_y = np.ones(self.Ny - 1)
        Dy2 = sp.diags([off_y, main_y, off_y], [-1, 0, 1]).tolil()
        Dy2[0, -1] = 1.0; Dy2[-1, 0] = 1.0
        Dy2 = (Dy2 / (self.dx**2)).tocsc()

        return (sp.kron(Dx2, sp.eye(self.Ny)) + sp.kron(sp.eye(self.Nx), Dy2)).tocsc()

    def solve_allen_cahn(self, eta_i, explicit_df, L, kappa, dt):
        # (I - dt * L * kappa * Lap) * eta_new = eta_old - dt * L * explicit_df
        A = self.Eye - (dt * L * kappa) * self.Lap
        b = eta_i.flatten() - (dt * L) * explicit_df.flatten()
        return spla.spsolve(A, b).reshape(self.Nx, self.Ny)