# burgers_io.py
import os
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import Dataset

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------

def save_burgers_dataset(ds, path: str, name: str = "burgers"):
    """
    Persist a BurgerEquationDatasetFNO to disk as a .pt file.

    Parameters
    ----------
    ds   : BurgerEquationDatasetFNO
        The dataset object you want to freeze.
    path : str
        Folder where the file will be written (created if missing).
    name : str
        Prefix of the file; date stamp is appended automatically.

    Returns
    -------
    str : full path of the stored .pt file.
    """
    os.makedirs(path, exist_ok=True)

    fname = f"{name}-date-{datetime.now().strftime('%Y-%m-%d')}.pt"
    full_path = os.path.join(path, fname)

    torch.save(
        {
            # big tensors -----------------------------------------------------------------
            "forcings" : torch.from_numpy(ds.forcings),            # (N, Nx, Nt)
            "targets"  : torch.from_numpy(ds.targets),             # (N, Nx, Nt)
            "mask"     : torch.from_numpy(ds.supervised_mask),     # (N,)
            "nu_vals"  : torch.from_numpy(ds.nu_vals),             # (N,)

            # 1‑D grids (float32 keeps file small) -----------------------------------------
            "x" : torch.tensor(ds.x, dtype=torch.float32),         # (Nx,)
            "t" : torch.tensor(ds.t, dtype=torch.float32),         # (Nt,)

            # a little meta info – handy for reproducibility -------------------------------
            "meta" : dict(
                Nx           = ds.Nx,
                Nt           = ds.Nt,
                nu_range     = (ds.nu_min, ds.nu_max),
                domain       = (float(ds.x.min()), float(ds.x.max())),
                time         = (float(ds.t.min()), float(ds.t.max())),
                function_types   = tuple(ds.function_types),
                include_supervision = bool(ds.include_supervision),
                fraction_supervised = float(ds.fraction_supervised),
                project      = bool(ds.project),
                bound        = tuple(ds.bound),
                add_noise    = bool(ds.add_noise),
            ),
        },
        full_path,
    )

    print(f"[burgers_io] Saved dataset -> {full_path}")
    return full_path


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

class _CachedBurgersDataset(Dataset):
    """
    A lightweight Dataset that wraps tensors saved by `save_burgers_dataset`.
    """
    def __init__(self, saved_dict):
        # core tensors (left on CPU; DataLoader can pin‑memory / push to GPU)
        self.f = saved_dict["forcings"].float()        # (N, Nx, Nt)
        self.u = saved_dict["targets"].float()         # (N, Nx, Nt)
        self.mask = saved_dict["mask"].bool()          # (N,)
        self.nu_vals = saved_dict["nu_vals"].float()   # (N,)

        # 1‑D grids
        self.x = saved_dict["x"].float()               # (Nx,)
        self.t = saved_dict["t"].float()               # (Nt,)

        # pre‑make meshgrid once to avoid re‑allocations in __getitem__
        X, T = torch.meshgrid(self.x, self.t, indexing="ij")
        self.X = X                                   # (Nx, Nt)
        self.T = T                                   # (Nx, Nt)

    # ---- PyTorch Dataset API ----
    def __len__(self):
        return self.f.shape[0]

    def __getitem__(self, idx):
        f      = self.f[idx]              # (Nx, Nt)
        target = self.u[idx]              # (Nx, Nt)
        m      = self.mask[idx]           # bool
        nu     = self.nu_vals[idx]        # scalar

        # Build 4‑channel input exactly as in original dataset:
        log_nu = torch.full_like(f, torch.log(nu))
        inp    = torch.stack([f, self.X, self.T, log_nu], dim=-1)   # (Nx, Nt, 4)

        return inp, target.unsqueeze(-1), m, nu


def load_burgers_dataset(pt_file: str) -> Dataset:
    """
    Load the .pt file produced by `save_burgers_dataset` and get a PyTorch dataset.

    Example
    -------
    >>> ds_cached = load_burgers_dataset("datasets/burgers/burgers-date-2025-07-25.pt")
    >>> dl = DataLoader(ds_cached, batch_size=64, shuffle=True, collate_fn=custom_collate_fno_fn)
    """
    saved = torch.load(pt_file, map_location="cpu")
    print(f"[burgers_io] Loaded dataset <- {pt_file}")
    return _CachedBurgersDataset(saved)

def solve_burgers_equation(
    f, x, t, *, nu=0.01, cfl=0.4, dtype=np.float64, bc="dirichlet"
):
    """
    Finite‑difference Burgers solver with **adaptive time step**:
        dt := min(cfl*dx**2/(2*nu),  cfl*dx/max|u|)

    Parameters
    ----------
    f   : (Nx, Nt) array, forcing term (already on same grid)
    x,t : 1‑D grids (len = Nx, Nt)
    nu  : viscosity
    cfl : safety factor (0<cfl<=1)
    dtype: float precision. Use float64 to push overflow farther away.
    bc  : 'dirichlet' | 'periodic'
    """
    Nx, Nt = len(x), len(t)
    dx     = x[1] - x[0]
    dt_grid= t[1] - t[0]          # spacing of the *given* t‑grid

    u  = np.zeros((Nx, Nt), dtype=dtype)
    dt = dt_grid                  # start with the grid spacing

    for n in range(Nt-1):
        #-- stability limits -------------------------------------------------
        umax   = np.max(np.abs(u[:, n]))
        dt_diff= dx*dx / (2*nu + 1e-12)     # diffusion limit
        dt_conv= dx / (umax + 1e-8)         # convection limit
        dt     = min(dt, cfl*dt_diff, cfl*dt_conv)

        #-- explicit FT‑CS update -------------------------------------------
        u_next = u[:, n].copy()

        # interior points
        du_dx     = (u[2:, n] - u[:-2, n]) / (2*dx)
        d2u_dx2   = (u[2:, n] - 2*u[1:-1, n] + u[:-2, n]) / dx**2
        u_next[1:-1] = (
            u[1:-1, n]
            - dt * u[1:-1, n] * du_dx
            + dt * nu * d2u_dx2
            + dt * f[1:-1, n]
        )

        # boundary conditions
        if bc == "dirichlet":
            u_next[ 0] = 0.0
            u_next[-1] = 0.0
        elif bc == "periodic":
            u_next[ 0] = u_next[-2]
            u_next[-1] = u_next[ 1]

        u[:, n+1] = u_next

        # optional: overwrite the next‑step Δt in the stored t‑grid
        t[n+1] = t[n] + dt

    return u.astype(np.float32)   # cast back for training

# -----------------------------------------------------------------------------
# Gaussian random field generator (for forcing term)
# -----------------------------------------------------------------------------

def generate_gaussian_random_field_2d(Nx, Nt, length_scale_x=0.2, length_scale_t=0.2, variance=1.0):
    kx = np.fft.fftfreq(Nx, d=1.0 / Nx)
    kt = np.fft.fftfreq(Nt, d=1.0 / Nt)
    KX, KT = np.meshgrid(kx, kt, indexing="ij")
    K2 = (KX ** 2) / (length_scale_x ** 2) + (KT ** 2) / (length_scale_t ** 2)
    psd = variance * np.exp(-K2)
    real = np.random.normal(size=(Nx, Nt))
    imag = np.random.normal(size=(Nx, Nt))
    spectrum = np.sqrt(psd) * (real + 1j * imag)
    grf = np.fft.ifft2(spectrum).real
    grf -= grf.mean()
    std = grf.std() if grf.std() > 1e-8 else 1.0
    grf *= np.sqrt(variance) / std
    return grf

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class BurgerEquationDatasetFNO(Dataset):
    """Physics‑informed dataset for Burgers control forcing → state mapping."""

    def __init__(
        self,
        n_samples: int,
        Nx: int,
        Nt: int,
        nu_range=(0.01, 0.05),
        domain=(0.0, 1.0),
        time=(0.0, 1.0),
        function_types=("sine", "grf"),
        include_supervision=False,
        fraction_supervised=0.2,
        project=True,
        bound=(-3, 3),
        add_noise=True,
    ):
        super().__init__()
        self.Nx = Nx
        self.Nt = Nt
        self.include_supervision = include_supervision
        self.fraction_supervised = fraction_supervised
        self.function_types = function_types
        self.project = project
        self.bound = bound
        self.add_noise = add_noise
        self.nu_min, self.nu_max = nu_range

        # Grids
        self.x = np.linspace(*domain, Nx)
        self.t = np.linspace(*time, Nt)
        self.X, self.T = np.meshgrid(self.x, self.t, indexing="ij")  # (Nx,Nt)

        self.forcings = []  # f(x,t)
        self.targets = []   # u(x,t)
        self.supervised_mask = []
        self.nu_vals = []   # scalar ν per sample

        for _ in range(n_samples):
            func_type = np.random.choice(self.function_types)
            f = self._generate_control(func_type)
            if self.project:
                f = self._project_to_range(f)
            if self.add_noise:
                f += np.random.normal(0.0, 0.1, size=f.shape)

            # draw ν for this sample (log‑uniform in practise → here uniform)
            nu = np.random.uniform(self.nu_min, self.nu_max)

            if self.include_supervision and np.random.rand() < self.fraction_supervised:
                u = solve_burgers_equation(f, self.x, self.t, nu=nu)
                self.supervised_mask.append(True)
            else:
                u = np.zeros_like(f)
                self.supervised_mask.append(False)

            self.forcings.append(f)
            self.targets.append(u)
            self.nu_vals.append(nu)

        # to arrays for faster indexing
        self.forcings = np.stack(self.forcings)
        self.targets = np.stack(self.targets)
        self.supervised_mask = np.array(self.supervised_mask)
        self.nu_vals = np.array(self.nu_vals)

    # ------------------------------------------------------------------
    # helper functions
    # ------------------------------------------------------------------

    def _generate_control(self, func_type: str):
        if func_type == 'sine':
            A = np.random.uniform(0.5, 3.0)
            fx = np.random.uniform(1, 5)
            ft = np.random.uniform(0.5, 5.0)
            phix = np.random.uniform(0, 2*np.pi)
            phit = np.random.uniform(0, 2*np.pi)
            return A * np.sin(2 * np.pi * fx * self.X + phix) * np.sin(2 * np.pi * ft * self.T + phit)

        elif func_type == 'grf':
            return generate_gaussian_random_field_2d(
                self.Nx, self.Nt,
                length_scale_x=np.random.uniform(0.03, 0.3),
                length_scale_t=np.random.uniform(0.03, 0.3)
            )

        elif func_type == 'step':
            x_step = np.random.randint(1, self.Nx - 1)
            t_step = np.random.randint(1, self.Nt - 1)
            val = np.random.uniform(-2, 2)
            f = np.zeros((self.Nx, self.Nt))
            f[x_step:, t_step:] = val
            return f

        elif func_type == 'poly':
            c0 = np.random.uniform(-1, 1)
            c1 = np.random.uniform(-2, 2)
            c2 = np.random.uniform(-1, 1)
            c3 = np.random.uniform(-2, 2)
            c4 = np.random.uniform(-1, 1)
            return c0 + c1 * self.X + c2 * self.X**2 + c3 * self.T + c4 * self.T**2

        elif func_type == 'linear':
            slope_x = np.random.uniform(-2, 2)
            slope_t = np.random.uniform(-2, 2)
            return slope_x * self.X + slope_t * self.T

        elif func_type == 'bump':
            center_x = np.random.uniform(0.2, 0.8)
            center_t = np.random.uniform(0.2, 0.8)
            sigma_x = np.random.uniform(0.01, 0.1)
            sigma_t = np.random.uniform(0.01, 0.1)
            bump = np.exp(-((self.X - center_x)**2 / (2 * sigma_x**2) + (self.T - center_t)**2 / (2 * sigma_t**2)))
            return np.random.uniform(1.0, 3.0) * bump

        elif func_type == 'fourier':
            f = np.zeros_like(self.X)
            n_terms = np.random.randint(2, 6)
            for _ in range(n_terms):
                kx = np.random.randint(1, 5)
                kt = np.random.randint(1, 5)
                A = np.random.uniform(0.5, 2.0)
                phase_x = np.random.uniform(0, 2 * np.pi)
                phase_t = np.random.uniform(0, 2 * np.pi)
                f += A * np.sin(2 * np.pi * kx * self.X + phase_x) * np.sin(2 * np.pi * kt * self.T + phase_t)
            return f

        elif func_type == 'mixed':
            subtypes = ['sine', 'grf', 'step', 'poly', 'linear', 'bump', 'fourier']
            components = [self._generate_control(np.random.choice(subtypes)) for _ in range(np.random.randint(2, 4))]
            weights = np.random.uniform(0.3, 1.0, size=len(components))
            return sum(w * c for w, c in zip(weights, components))

    def _project_to_range(self, f):
        umin, umax = self.bound
        f_min, f_max = f.min(), f.max()
        if np.isclose(f_max, f_min):
            return np.random.uniform(umin, umax, size=f.shape)
        return (f - f_min) / (f_max - f_min) * (umax - umin) + umin

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.forcings)

    def __getitem__(self, idx):
        f = self.forcings[idx]
        u = self.targets[idx]
        nu = self.nu_vals[idx]
        mask = self.supervised_mask[idx]

        # build 4‑channel input (f, x, t, log ν)
        log_nu_channel = np.full_like(f, np.log(nu))
        input_tensor = np.stack([f, self.X, self.T, log_nu_channel], axis=-1)  # (Nx,Nt,4)

        return input_tensor, u[..., None], mask, nu  # shapes: (Nx,Nt,4), (Nx,Nt,1)

# -----------------------------------------------------------------------------
# DataLoader collate
# -----------------------------------------------------------------------------

def custom_collate_fno_fn(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, targets, masks, nus = zip(*batch)
    inputs = torch.tensor(np.stack(inputs), dtype=torch.float32, device=device)
    targets = torch.tensor(np.stack(targets), dtype=torch.float32, device=device)
    masks = torch.tensor(masks, dtype=torch.bool, device=device)
    nus = torch.tensor(nus, dtype=torch.float32, device=device)  # (B,)
    return inputs, targets, masks, nus
