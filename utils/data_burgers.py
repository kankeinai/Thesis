import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import solve_ivp
import h5py


class Burgers1D(Dataset):
    """
    Dataset for 1D inviscid Burgers' equation with time-invariant control:
        ∂y/∂t + y ∂y/∂x = u(x)
    """
    def __init__(
        self,
        n_samples: int,
        Nx: int,
        Nt: int,
        control_functions = ["grf", "sine", "fourier"],
        domain_x=(0.0, 1.0),
        domain_t=(0.0, 1.0),
        nu: float = 0.01,
        smoothing: bool = False,
        include_supervision: bool = True,
        fraction_supervised: float = 1.0,
        project: bool = False,
        bounds=(-3, 3),
        solver="rk4",
        save_path: str = None,
        name = "train"
    ):
        self.n_samples = n_samples
        self.Nx = Nx
        self.Nt = Nt
        self.control_functions = control_functions
        self.include_supervision = include_supervision
        self.fraction_supervised = fraction_supervised
        self.project = project
        self.solver = solver

        if project:
            self.bounds = bounds

        self.smoothing = smoothing
        self.nu = nu

        self.x = np.linspace(*domain_x, Nx)
        self.t = np.linspace(*domain_t, Nt)
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]

        self.controls = []
        self.trajectories = []
        self.supervised_mask = []

        for _ in range(n_samples):
            control_type = np.random.choice(self.control_functions)
            u = self._generate_control(control_type)
            if self.project:
                u = self._project_to_range(u)
            
            if self.smoothing:
                u = gaussian_filter1d(u, sigma=2)

            supervised = self.include_supervision and np.random.rand() < self.fraction_supervised
            if supervised:
                y = self._solve_burgers(u)
                if y.shape != (self.Nx, self.Nt):
                    print(f"[Warning] Trajectory shape mismatch: got {y.shape}, expected ({self.Nx}, {self.Nt})")
                self.trajectories.append(y.astype(np.float32))
                self.supervised_mask.append(True)
            else:
                self.trajectories.append(np.zeros((Nx, Nt), dtype=np.float32))
                self.supervised_mask.append(False)

            self.controls.append(u.astype(np.float32))

        self.controls = np.stack(self.controls)
        self.trajectories = np.stack(self.trajectories)
        self.supervised_mask = np.array(self.supervised_mask)

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fname = f"burgers_1d_dataset_{name}_{datetime.now().strftime('%Y-%m-%d')}.h5"
            full_path = os.path.join(save_path, fname)
            with h5py.File(full_path, "w") as f:
                f.create_dataset("controls", data=self.controls)
                f.create_dataset("trajectories", data=self.trajectories)
                f.create_dataset("mask", data=self.supervised_mask)
                f.create_dataset("x", data=self.x)
                f.create_dataset("t", data=self.t)
            print(f"[Burgers1D] Saved dataset -> {full_path}")

    def _solve_burgers(self, u):
            if self.solver == "rk4":
                return self._solve_burgers_rk4(u)
            elif self.solver == "ivp":
                return self._solve_burgers_ivp(u)
            else:
                raise ValueError(f"Unknown solver: {self.solver}")

    def _solve_burgers_rk4(self, u):
        def rhs(y):
            dy_dx = np.zeros_like(y)
            d2y_dx2 = np.zeros_like(y)

            dy_dx[1:-1] = (y[2:] - y[:-2]) / (2 * self.dx)
            dy_dx[0] = (y[1] - y[0]) / self.dx
            dy_dx[-1] = (y[-1] - y[-2]) / self.dx

            d2y_dx2[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / self.dx**2
            d2y_dx2[0] = d2y_dx2[1]
            d2y_dx2[-1] = d2y_dx2[-2]

            return -y * dy_dx + self.nu * d2y_dx2 + u

        y = np.full(self.Nx, 0.1)
        traj = np.zeros((self.Nx, self.Nt))
        traj[:, 0] = y

       
        for n in range(1, self.Nt):
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * self.dt * k1)
            k3 = rhs(y + 0.5 * self.dt * k2)
            k4 = rhs(y + self.dt * k3)
            y = y + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            traj[:, n] = y

        return traj

    def _solve_burgers_ivp(self, u):
        def rhs(t, y):
            dy_dx = np.zeros_like(y)
            d2y_dx2 = np.zeros_like(y)

            dy_dx[1:-1] = (y[2:] - y[:-2]) / (2 * self.dx)
            dy_dx[0] = (y[1] - y[0]) / self.dx
            dy_dx[-1] = (y[-1] - y[-2]) / self.dx

            d2y_dx2[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / self.dx**2
            d2y_dx2[0] = d2y_dx2[1]
            d2y_dx2[-1] = d2y_dx2[-2]

            return -y * dy_dx + self.nu * d2y_dx2 + u

        y0 = np.full(self.Nx, 0.1)
        sol = solve_ivp(rhs, [self.t[0], self.t[-1]], y0,
                        method="LSODA", dense_output=True,
                        max_step=self.dt)

        if not sol.success:
            raise RuntimeError(sol.message)

        return sol.sol(self.t)


    def _generate_control(self, control_type):
        if control_type == 'sine':
            A = np.random.uniform(0.5, 6.0)
            f = np.random.uniform(1, 5)
            phi = np.random.uniform(0, 2 * np.pi)
            return A * np.sin(2 * np.pi * f * self.x + phi)
        elif control_type == 'grf':
            u = self._generate_gaussian_random_field_fixed_1d(self.Nx, length_scale=0.2)
            return gaussian_filter1d(u, sigma=2) if self.smoothing else u
        elif control_type == 'fourier':
            u = np.zeros_like(self.x)
            for _ in range(np.random.randint(2, 5)):
                k = np.random.randint(1, 6)
                A = np.random.uniform(0.5, 3.0)
                phi = np.random.uniform(0, 2 * np.pi)
                u += A * np.sin(2 * np.pi * k * self.x + phi)
            return u
        elif control_type == 'step':
            x0 = np.random.uniform(0.2, 0.8)
            height = np.random.uniform(-2.0, 2.0)
            return height * (self.x > x0).astype(float)
        else:
            raise ValueError(f"Unknown control type: {control_type}")

    def _generate_gaussian_random_field_fixed_1d(self, grid_size, length_scale=0.2, mean=0, variance=1):
        kx = np.fft.fftfreq(grid_size)
        k_squared = kx**2
        k_squared[0] = np.inf
        psd = variance * np.exp(-k_squared * (length_scale**2))
        random_field = np.fft.ifft(np.sqrt(psd) * (np.random.normal(size=grid_size) + 1j * np.random.normal(size=grid_size)))
        field = np.real(random_field)
        return mean + (field - np.mean(field)) * (np.sqrt(variance) / np.std(field))

    def _project_to_range(self, u, ):
        umin, umax = self.bounds
        u_min, u_max = u.min(), u.max()
        if np.isclose(u_max, u_min):
            return np.random.uniform(umin, umax, size=u.shape)
        return (u - u_min) / (u_max - u_min) * (umax - umin) + umin
    

def load_burgers1d_dataset(filepath):
    """
    Load 1D heat equation dataset from HDF5 file and return a PyTorch Dataset object.
    """
    import h5py
    class Burgers1DLoadedDataset(torch.utils.data.Dataset):
        def __init__(self, path):
            with h5py.File(path, "r") as f:
                self.controls = torch.tensor(f["controls"][:], dtype=torch.float32)    # (N, Nx)
                self.trajectories = torch.tensor(f["trajectories"][:], dtype=torch.float32)  # (N, Nx, Nt)
                self.mask = torch.tensor(f["mask"][:], dtype=torch.bool)
                self.x = torch.tensor(f["x"][:], dtype=torch.float32)
                self.t = torch.tensor(f["t"][:], dtype=torch.float32)

        def __len__(self):
            return self.controls.shape[0]

        def __getitem__(self, idx):
            return self.controls[idx], self.trajectories[idx], self.mask[idx]

    return  Burgers1DLoadedDataset(filepath)


def custom_collate_fno1d_fn(batch):
    """
    Collate function to batch control and trajectory samples for FNO1d.
    """
    u_list, y_list, m_list = zip(*batch)  # each u: (Nx,), y: (Nx, Nt)
    u_tensor = torch.stack(u_list)        # (B, Nx)
    x_coords = torch.linspace(0, 1, u_tensor.shape[1])
    x_coords = x_coords.unsqueeze(0).repeat(u_tensor.shape[0], 1)
    x_in = torch.stack([u_tensor, x_coords], dim=-1)  # (B, Nx, 2)
    y_tensor = torch.stack(y_list).permute(0, 1, 2)    # (B, Nx, Nt)
    m_tensor = torch.tensor(m_list).bool()
    return x_in, y_tensor, m_tensor

