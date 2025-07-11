import torch
from datetime import datetime
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from functorch import grad, vmap
from torch.func import jacrev  # or use autograd.functional.jacobian
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from scipy.signal import savgol_filter

import torch
import numpy as np


import torch

import torch
import torch.nn.functional as F
from typing import Tuple

import torch
from typing import Tuple

def cubic_spline_interp(
    x: torch.Tensor,    # shape [B, N]
    t: torch.Tensor     # shape [M], values in [0,1]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a batch of B signals x on a uniform N-point [0,1] grid,
    and M query-points t (shared across batch), compute:
      - out: the natural cubic-spline interpolant S(t) at each t[j]
      - dx_dt: the derivative S'(t) at each t[j]
    Returns (out, dx_dt), both of shape [B, M].
    """
    B, N = x.shape
    M = t.shape[0]
    device, dtype = x.device, x.dtype
    h = 1.0 / (N - 1)

    # --- 1) solve for second derivatives m at the N knots ---
    # right-hand side size [B, N-2]
    d = 6.0 * (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]) / (h*h)

    k = N - 2
    a = torch.ones(k-1, device=device, dtype=dtype)
    b = 4.0 * torch.ones(k,   device=device, dtype=dtype)
    c = torch.ones(k-1, device=device, dtype=dtype)

    # Thomas algorithm
    cp = torch.empty(k-1, device=device, dtype=dtype)
    dp = torch.empty(B, k,   device=device, dtype=dtype)
    cp[0]   = c[0] / b[0]
    dp[:,0] = d[:,0] / b[0]
    for i in range(1, k-1):
        denom   = b[i] - a[i-1] * cp[i-1]
        cp[i]   = c[i] / denom
        dp[:,i] = (d[:,i] - a[i-1] * dp[:,i-1]) / denom
    dp[:,k-1] = (d[:,k-1] - a[k-2] * dp[:,k-2]) / (b[k-1] - a[k-2] * cp[k-2])

    m_int = torch.empty_like(dp)
    m_int[:,-1] = dp[:,-1]
    for i in range(k-2, -1, -1):
        m_int[:,i] = dp[:,i] - cp[i] * m_int[:,i+1]

    # full second-derivative array, with natural BCs (zero at ends)
    m = torch.zeros(B, N, device=device, dtype=dtype)
    m[:,1:-1] = m_int

    # --- 2) for each query t, find its interval [t_i, t_{i+1}] ---
    # lift to [B, M, 1] so we can do batch-gather below
    t_full = t.view(1, M, 1).expand(B, M, 1)

    # continuous index u = t*(N-1)
    u  = t_full * (N - 1)
    i0 = u.floor().long().clamp(0, N-2)  # left knot index, shape [B,M,1]
    i1 = i0 + 1                         # right knot

    # distances from each knot
    t_i  = i0.to(dtype) * h              # shape [B,M,1]
    dt0  = (t_i + h - t_full)            # = t_{i+1} - t
    dt1  = (t_full - t_i)                # = t - t_i

    # --- 3) gather y_i, y_{i+1}, m_i, m_{i+1} ---
    # after squeeze, shapes [B,M]
    y0 = x.gather(1, i0.squeeze(-1))
    y1 = x.gather(1, i1.squeeze(-1))
    m0 = m.gather(1, i0.squeeze(-1))
    m1 = m.gather(1, i1.squeeze(-1))

    # --- 4) compute spline and its derivative via Hermite form ---
    coeff0 = m0 / (6 * h)
    coeff1 = m1 / (6 * h)

    S = (
        coeff0 * dt0.squeeze(-1)**3 +
        coeff1 * dt1.squeeze(-1)**3 +
        (y0 - m0*h*h/6) * (dt0.squeeze(-1)/h) +
        (y1 - m1*h*h/6) * (dt1.squeeze(-1)/h)
    )   # [B, M]

    dS_dt = (
        -m0/(2*h) * dt0.squeeze(-1)**2 +
         m1/(2*h) * dt1.squeeze(-1)**2 +
        -(y0 - m0*h*h/6)/h +
         (y1 - m1*h*h/6)/h
    )   # [B, M]

    return S, dS_dt


def compute_loss_nde(model, u, t, t0, ut, t_grid, method="finite"):

    device = u.device
    t_vec = t_grid[0, :, 0].to(device)
    x3 = model(u, t_grid)         # → (B, N, 1)
    x  = x3.squeeze(-1)      # →
    initial = (x[:, 0] - 1.0).pow(2).mean()

    if method == "finite":
        # central differences + one‐sided at boundaries
        dx_dt,  = torch.gradient(x, spacing=(t_vec,), dim=1)
        u0 = u.squeeze(-1)  # (B, N)
        residual = dx_dt - 5/2 * (-x + x*u0 + u0**2)

    elif method == "interpolate":

        x_new, dx_dt = cubic_spline_interp(x, t)
        u0 = ut.squeeze(-1)
        residual = dx_dt - 5/2 * (-x_new + x_new*u0 + u0**2)

    # 4) Losses
    physics_loss = residual.pow(2).mean()


    return physics_loss, initial


def compute_loss_ode(model, u, t, t0, ut, t_grid, method="finite"):
    """
    Compute the physics‐informed and initial‐condition loss for
        x'(t) + x(t) - u(t) = 0

    Parameters
    ----------
    model  : nn.Module
        Maps (u, t) -> x of shape (B, N, 1).
    u      : torch.Tensor, shape (B, N, 1)
        Forcing term.
    t      : torch.Tensor, shape (B, N, 1)
        Time grid (requires_grad only if method="autograd").
    method : {"finite", "autograd", "spectral"}
        How to compute ∂x/∂t.

    Returns
    -------
    physics_loss : torch.Tensor
    initial_loss : torch.Tensor
    """

    # 1) Single forward pass
    
    device = u.device
    t_vec = t_grid[0, :, 0].to(device)
    x3 = model(u, t_grid)         # → (B, N, 1)
    x  = x3.squeeze(-1)      # → (B, N)

    initial_loss = (x[:, 0] - 1.0).pow(2).mean()

    # 3) Compute dx/dt
    if method == "finite":
        # central differences + one‐sided at boundaries
        dx_dt,  = torch.gradient(x, spacing=(t_vec,), dim=1)
        u0 = u.squeeze(-1)  # (B, N)
        residual = dx_dt + x - u0

    elif method == "interpolate":

        x_new, dx_dt = cubic_spline_interp(x, t)
        ut = ut.squeeze(-1)
        residual = dx_dt + x_new - ut

    else:
        raise ValueError(f"Unknown method {method!r}")
    
    physics_loss = residual.pow(2).mean()

    return physics_loss, initial_loss


def train_fno(model, compute_loss, dataloader, optimizer, scheduler, epochs, t_grid, w=[1,1], method="autograd", folder = "trained_models/fno", logging = True):
    """
    Train a Fourier Neural Operator (FNO) model.

    Parameters
    ----------
    model : nn.Module
        The FNO model to be trained.
    dataloader : DataLoader
        PyTorch DataLoader providing batches of input fields `u`.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler to step after each epoch.
    epochs : int
        Number of full passes over the training dataset.
    t_grid : torch.Tensor
        Precomputed time or coordinate grid of shape (batch_max, N, 1).
    w : list of two floats, optional
        Loss weights `[w_phys, w_init]` for physics-based and initial-condition losses.
    folder : str, optional
        Directory path where model checkpoints will be saved.
    logging : bool, optional
        If True, prints batch-level loss information during training.

    Returns
    -------
    model : nn.Module
        The trained model (state_dict saved to disk each epoch).
    """
      # (N,)
    for epoch in range(epochs):

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        start_time = datetime.now()

        for u, t, t0, ut in dataloader:
            
            u = u.to(device)

     
            optimizer.zero_grad()
            physics_loss, initial_loss = compute_loss(model, u, t, t0, ut, t_grid, method=method)
            loss = w[0]*physics_loss + w[1]*initial_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if logging:
                print(f'Epoch [{epoch+1}/{epochs}], Physics loss: {physics_loss}, Initial loss: {initial_loss}')

        
        epoch_loss /= n_batches
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, Time: {(datetime.now() - start_time).total_seconds()} s') 
              
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        os.makedirs(folder, exist_ok=True)
        torch.save(model.state_dict(), folder+f"/epoch-[{epoch+1}]_model_{timestamp}_loss{epoch_loss:.4f}.pth")
        scheduler.step()

    return model

def train_lno(model, compute_loss, dataloader, optimizer, scheduler, epochs, t_grid, method="autograd",save=10, w = [1,1], folder = "trained_models/lno", logging = True):
    """
    Train a Lagrangian Neural Operator (LNO) model.

    Parameters
    ----------
    model : nn.Module
        The LNO model to be trained.
    dataloader : DataLoader
        PyTorch DataLoader providing batches of (u, t) inputs.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler to step after each epoch.
    epochs : int
        Number of epochs (full passes over the dataset).
    w : list of two floats, optional
        Loss weights `[w_phys, w_init]` for physics-based and initial-condition losses.
    folder : str, optional
        Directory path where model checkpoints will be saved every 10 epochs.
    logging : bool, optional
        If True, prints batch-level loss information during training.

    Returns
    -------
    None
        The model is trained in-place and checkpoints are saved to disk.
    """
    for epoch in range(epochs):
        
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        start_time = datetime.now()


        for u, _, _, _ in dataloader:

            u = u.to(device).unsqueeze(-1)
            t  = t_grid.repeat([u.shape[0], 1, 1]).clone().detach().requires_grad_(True)


            physics_loss, initial_loss = compute_loss(model, u, t, method=method)
            loss = w[0] * physics_loss + w[1] * initial_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if logging:
                print(f"Epoch: {epoch}, Physics loss: {physics_loss}, Initial loss: {initial_loss}")

        epoch_loss /= n_batches

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, Time: {(datetime.now() - start_time).total_seconds()} s') 

        if (epoch + 1) % save == 0:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            os.makedirs(folder, exist_ok=True)
            model_filename = f'epochs_[{epoch+1}]_model_time_[{timestamp}]_loss_[{epoch_loss:.4f}].pth'
            torch.save(model.state_dict(), folder+f"/{model_filename}")

        scheduler.step()

def objective_function_ode(args, method="finite"):

    """
    Compute the total objective J = w_control * control_cost
    + w_phys * physics_loss + w_init * initial_loss.

    Parameters
    ----------
    args : dict
        'x' : torch.Tensor, shape (batch, N, 1) or (batch, N)  model output
        'u' : torch.Tensor, shape (batch, N, 1) or (batch, N)  control signal
        't' : torch.Tensor, shape (N,) or (batch, N, 1)  time grid
        'w' : list of three floats [w_control, w_phys, w_init]
    model_name : str
        'lno' or 'fno' determines how dt and residuals are computed.

    Returns
    -------
    J : torch.Tensor
        Scalar objective value.
    """

    u = args['u']
    t = args['t']
    w = args['w']
    x = args['x']

    u0 = u.squeeze(-1)            # (B, N)
    dx_dt = torch.zeros_like(x)

    if method == "finite":
        # pick out a single copy of the time‐vector, shape (N,)
        t_vec = t[0, :, 0]             # -> (N,)

        # compute the finite‐difference gradient along dim=1
        dx_dt = torch.gradient(
            x,
            spacing=(t_vec,),  # <-- a 1-D tensor of length N
            dim=1
        )[0]                        # -> (B, N)
    else:
        raise ValueError(f"Unknown method {method!r}")


    residual = dx_dt + x - u0

    physics_loss = torch.mean(residual ** 2)
    initial_loss = torch.mean((x[:, 0] - torch.ones(10, device = device))**2)
    control_cost = torch.mean(torch.trapz((x**2 + u**2).squeeze(), t.squeeze()))

    J = w[0] * control_cost + w[1] * physics_loss + w[2] * initial_loss 
    return J

def objective_function_nde(args, method="finite"):
    """
    J = w[0]*control_cost + w[1]*physics_loss + w[2]*boundary_penalty
    enforcing u ∈ [u_min, u_max] by reparameterization.
    """
    x = args['x']
    u = args['u']
    t  = args['t']
    w = args['w']

    # 2) compute dx/dt via finite‐differences
    if method == "finite":
        t_vec = t[0, :, 0]         # (N,)
        dx_dt, = torch.gradient(x, spacing=(t_vec,), dim=1)
    else:
        raise ValueError(f"Unknown method {method!r}")

    # 3) physics residual with bounded u
    #    dx/dt + (5/9)*( -x + x*u + u**2 )  = 0
    residual = dx_dt - (5/2)*(-x + x*u - u**2)
    physics_loss = residual.pow(2).mean()
    initial_loss = (x[:, 0] - 1.0).pow(2)

    # 4) end‐point control cost (e.g. minimize x(T)^2)
    control_cost = -x[:, -1]
    du_dt, = torch.gradient(u, spacing=(t_vec,), dim=1)
    d2u_dt2, = torch.gradient(du_dt, spacing=(t_vec,), dim=1)

    noise_cost = du_dt.pow(2).mean()

    # 6) total objective
    J = w[0] * control_cost \
      + w[1] * physics_loss \
      + w[2] * initial_loss \
      + w[3] * noise_cost

    return J


def optimize_neural_operator(model, objective_function, m, end_time, num_epochs, learning_rate, bounds=1, w=[1,1,1], model_name="lno"):  

    if model_name=="lno":
        u = torch.randn(1, m, 1, device=device, requires_grad=True)
        t = torch.tensor(torch.linspace(0, end_time, m, ), dtype=torch.float).reshape(1, m, 1).to(device)
    elif model_name=="fno":
        u = torch.randn(1, m, device=device, requires_grad=True)
        t = torch.linspace(0, end_time, m, device=device).unsqueeze(0).unsqueeze(-1)  # shape: [1, m, 1]

    optimizer = torch.optim.Adam([u], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    for epoch in range(num_epochs):

        optimizer.zero_grad()
        
        x = model(u, t)

        args = {'u': u, 'x': x, 't': t, 'w':w}

        loss = objective_function(args)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, time: {datetime.now().time()}')

        scheduler.step()
    
    x = model(u, t).squeeze(-1)  
    t = t.squeeze(-1)            # shape: [1, m]

    return u, x, t

def plot_analytical_and_found_solutions(t_plot, analytical_x_vals, x_found, analytical_u_vals, u_found):

    """
    Plot analytical vs. optimized solutions for state x and control u.

    Parameters
    ----------
    t_plot : array-like
        Time grid used for plotting.
    analytical_x_vals : array-like
        Analytical solution x(t) values.
    x_found : array-like
        Learned/optimized x values.
    analytical_u_vals : array-like
        Analytical control u(t) values.
    u_found : array-like
        Learned/optimized u values.
    """
    
    sns.set_theme(style="whitegrid", font_scale=1.2)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each series with consistent line widths and styles
    sns.lineplot(x=t_plot, y=analytical_x_vals, ax=ax, label="Analytical x", linewidth=2.5)
    sns.lineplot(x=t_plot, y=x_found,       ax=ax, label="Found x (FNO)",   linestyle="--", linewidth=2.5)
    sns.lineplot(x=t_plot, y=analytical_u_vals, ax=ax, label="Analytical u",  linewidth=2.5)
    sns.lineplot(x=t_plot, y=u_found,       ax=ax, label="Found u",          linestyle="--", linewidth=2.5)

    # Axis labels and bold title
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Value", fontsize=14)
    ax.set_title("Comparison of Analytical and Found Solutions", fontsize=16, fontweight="bold")

    # Tweak legend appearance
    ax.legend(frameon=True, edgecolor="gray", fontsize=12)

    plt.tight_layout()
    plt.show()