import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import solve_ivp
import h5py

class MultiFunctionDatasetPDE(Dataset):
    def __init__(
        self,
        n_samples: int,
        Nx: int,
        Nt: int,
        problem: str = "heat",
        properties: dict = {"nu": 0.01},
        control_functions =  ["grf",'sine', 'fourier'],
        domain_x=(0.0, 1.0),
        domain_t=(0.0, 1.0),
        smoothing: bool = False,
        include_supervision: bool = True,
        fraction_supervised: float = 1.0,
        project: bool = False,
        save_path: str = None,
        name = "train"
    ):
        if problem == "heat":
            try:
                self.nu = properties["nu"]
            except KeyError:
                raise ValueError("nu is required for heat equation")
        elif problem == "diffusion":
            try:
                self.diff_coef = properties["diff_coef"]
            except KeyError:
                raise ValueError("diff_coef is required for diffusion equation")
            try:
                self.reac_coef = properties["reac_coef"]
            except KeyError:
                raise ValueError("reac_coef is required for diffusion equation")
        elif problem == "burgers":
            try:
                self.nu = properties["nu"]
            except KeyError:
                raise ValueError("nu is required for burgers equation")
        else:
            raise ValueError(f"Unknown problem: {problem}")
        
        self.n_samples = n_samples
        self.Nx = Nx
        self.Nt = Nt
        self.control_functions = control_functions
        self.include_supervision = include_supervision
        self.fraction_supervised = fraction_supervised
        self.project = project
        self.smoothing = smoothing

        self.x = np.linspace(*domain_x, Nx)
        self.t = np.linspace(*domain_t, Nt)
        self.X, self.T = np.meshgrid(self.x, self.t, indexing="ij")

        self.controls = []
        self.trajectories = []
        self.supervised_mask = []

        for _ in range(n_samples):

            control_type = np.random.choice(self.control_functions)

            u = self._generate_control(control_type)

            if self.project:
                u = self._project_to_range(u)

            supervised = self.include_supervision and np.random.rand() < self.fraction_supervised

            if supervised:
                y = self._solve_pde(u)
                self.trajectories.append(y.astype(np.float32))
                self.supervised_mask.append(True)
            else:
                self.trajectories.append(np.zeros((Nx, Nt), dtype=np.float32))
                self.supervised_mask.append(False)

            self.controls.append(u.astype(np.float32))

        self.controls = np.stack(self.controls)
        self.trajectories = np.stack(self.trajectories)
        self.supervised_mask = np.array(self.supervised_mask)

        # Print dataset statistics
        u_amplitudes = self.controls.max(axis=1) - self.controls.min(axis=1)
        print("[HeatEquation1DControlDataset] Dataset Summary")
        print(f"  - Samples:             {self.n_samples}")
        print(f"  - Resolution:          Nx={self.Nx}, Nt={self.Nt}")
        print(f"  - Forcing types:        {self.control_functions}")
        print(f"  - Supervised samples:  {np.sum(self.supervised_mask)} / {self.n_samples}")
        print(f"  - Amplitude range:     {u_amplitudes.min():.2f} to {u_amplitudes.max():.2f}")
        print(f"  - Avg amplitude:       {u_amplitudes.mean():.2f} ± {u_amplitudes.std():.2f}")


        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fname = f"heat_1d_dataset_{name}_{datetime.now().strftime('%Y-%m-%d')}.h5"
            full_path = os.path.join(save_path, fname)
            with h5py.File(full_path, "w") as f:
                f.create_dataset("controls", data=self.controls)
                f.create_dataset("trajectories", data=self.trajectories)
                f.create_dataset("mask", data=self.supervised_mask)
                f.create_dataset("x", data=self.x)
                f.create_dataset("t", data=self.t)
            print(f"[heat1d_io] Saved dataset -> {full_path}")

    def _generate_control(self, control_type):
        if control_type == 'sine':
            A = np.random.uniform(0.5, 6.0)
            f = np.random.uniform(1, 5)
            phi = np.random.uniform(0, 2 * np.pi)
            u = A * np.sin(2 * np.pi * f * self.x + phi)
        elif control_type == 'grf':
            u = self._generate_gaussian_random_field_fixed_1d(self.Nx, length_scale=0.2)
            if self.smoothing:
                u = gaussian_filter1d(u, sigma=2)
        elif control_type == 'fourier':
            u = np.zeros_like(self.x)
            for _ in range(np.random.randint(2, 5)):
                k = np.random.randint(1, 6)
                A = np.random.uniform(0.5, 3.0)
                phi = np.random.uniform(0, 2 * np.pi)
                u += A * np.sin(2 * np.pi * k * self.x + phi)
        return u

    def _generate_gaussian_random_field_fixed_1d(self, grid_size, length_scale=0.2, mean=0, variance=1):
        kx = np.fft.fftfreq(grid_size)
        k_squared = kx**2
        k_squared[0] = np.inf
        psd = variance * np.exp(-k_squared * (length_scale**2))
        random_field = np.fft.ifft(np.sqrt(psd) * (np.random.normal(size=grid_size) + 1j * np.random.normal(size=grid_size)))
        field = np.real(random_field)
        field = mean + (field - np.mean(field)) * (np.sqrt(variance) / np.std(field))
        return field

    def _solve_pde(self, u):
        if self.problem == "heat":
            return self._solve_heat(u)
        elif self.problem == "diffusion":
            return self._solve_diffusion_reaction(u)
        else:
            raise ValueError(f"Solver not implemented for problem: {self.problem}")

    def _solve_heat(self, u):
        dx = self.x[1] - self.x[0]
        Nt = self.Nt
        Nx = self.Nx

        def rhs(t, y):
            y = y.reshape(Nx)
            d2y_dx2 = np.zeros_like(y)
            d2y_dx2[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / dx**2
            d2y_dx2[0] = 0
            d2y_dx2[-1] = 0
            return self.nu * d2y_dx2 + u

        y0 = np.zeros(Nx)
        sol = solve_ivp(rhs, [self.t[0], self.t[-1]], y0, t_eval=self.t, method='RK45')
        return sol.y
    
    def _solve_diffusion_reaction(self, u):
        dx = self.x[1] - self.x[0]
        Nx = self.Nx

        def rhs(t, y):
            y = y.reshape(-1)
            d2y_dx2 = np.zeros_like(y)
            
            # Second spatial derivative with Dirichlet BC (y(0)=0, y(1)=0)
            d2y_dx2[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / dx**2
            
            # Boundary conditions y(0)=0, y(1)=0 explicitly enforced
            d2y_dx2[0] = (0 - 2*y[0] + y[1]) / dx**2
            d2y_dx2[-1] = (y[-2] - 2*y[-1] + 0) / dx**2
            
            return self.diff_coef * d2y_dx2 - self.reac_coef * y**2 + u

        y0 = np.zeros(Nx)
        sol = solve_ivp(rhs, [self.t[0], self.t[-1]], y0, t_eval=self.t, method='Radau')
        return sol.y


    def _project_to_range(self, u, bound=(-3, 3)):
        umin, umax = bound
        u_min, u_max = u.min(), u.max()
        if np.isclose(u_max, u_min):
            return np.random.uniform(umin, umax, size=u.shape)
        return (u - u_min) / (u_max - u_min) * (umax - umin) + umin
    
def load_pde1d_dataset(filepath):
    """
    Load 1D heat equation dataset from HDF5 file and return a PyTorch Dataset object.
    """
    import h5py
    class PDE1DLoadedDataset(torch.utils.data.Dataset):
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

    return PDE1DLoadedDataset(filepath)


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
