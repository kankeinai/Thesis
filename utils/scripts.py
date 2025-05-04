import torch
from datetime import datetime
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from functorch import grad, vmap
from torch.func import jacrev  # or use autograd.functional.jacobian

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import torch
import numpy as np


import torch

def compute_loss_nde(model, u, t, method="finite"):
    # 1) Single forward pass

    u0 = u.squeeze(-1)       # → (B, N)
    B, N = u0.shape

    # 2) Time step
    t_vec = t[0, :, 0]               # (N,)
    x3 = model(u, t)         # → (B, N, 1)
    x  = x3.squeeze(-1)      # → (B, N)
    # central differences + one‐sided at boundaries
    dx_dt,  = torch.gradient(x, spacing=(t_vec,), dim=1)
       
    residual = dx_dt + 5/9 * (-x + x*u0 + u0**2)
    # 4) Losses
    physics_loss = residual.pow(2).mean()


    return physics_loss, 0


def compute_loss_ode(model, u, t, method="finite"):
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

    u0 = u.squeeze(-1)       # → (B, N)
    B, N = u0.shape

    # 2) Time step
    t_vec = t[0, :, 0]               # (N,)
    dt    = (t_vec[1] - t_vec[0]).item()

    # 3) Compute dx/dt
    if method == "finite":
        x3 = model(u, t)         # → (B, N, 1)
        x  = x3.squeeze(-1)      # → (B, N)
        # central differences + one‐sided at boundaries
        dx_dt,  = torch.gradient(x, spacing=(t_vec,), dim=1)
        
    elif method == "autograd":
       # vectorized “predict only”:
        x = vmap(lambda u_i, t_i: 
            model(u_i.unsqueeze(0), t_i.unsqueeze(0))
                .squeeze(0).squeeze(-1),
            in_dims=(0,0)
        )(u, t)    # → [B, N]

        # then your gradient code
        per_sample_jacs = vmap(jacrev(lambda u_i, t_i: 
            model(u_i.unsqueeze(0), t_i.unsqueeze(0))
                .squeeze(0).squeeze(-1)
        ), in_dims=(0,0))(u, t)  # → [B, N, N]

        idx   = torch.arange(N, device=u.device)
        dx_dt = per_sample_jacs[:, idx, idx]  # → [B, N]


    elif method == "spectral":
        x3 = model(u, t)         # → (B, N, 1)
        x  = x3.squeeze(-1)      # → (B, N)
        # FFT‐based derivative
        X     = torch.fft.rfft(x, dim=-1)                     
        freqs = torch.fft.rfftfreq(n=N, d=dt, device=x.device) 
        omega = 2 * torch.pi * freqs                          
        X_dt  = (1j * omega.unsqueeze(0)) * X                  
        dx_dt = torch.fft.irfft(X_dt, n=N, dim=-1)             


    else:
        raise ValueError(f"Unknown method {method!r}")
    residual = dx_dt + x - u0
    # 4) Losses
    physics_loss = residual.pow(2).mean()
    initial_loss = (x[:, 0] - 1.0).pow(2).mean()

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
    for epoch in range(epochs):

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        start_time = datetime.now()

        for u, _, _, _ in dataloader:

            u = u.to(device)
            t = t_grid[:u.size(0), :].to(device)
            t.requires_grad_(True)    # (only if t_grid itself is a leaf with requires_grad=False)

            optimizer.zero_grad()
            
            physics_loss, initial_loss = compute_loss(model, u, t, method=method)
            loss = w[0]*physics_loss + w[1]*initial_loss
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

            optimizer.zero_grad()

            physics_loss, initial_loss = compute_loss(model, u, t, method=method)
            loss = w[0] * physics_loss + w[1] * initial_loss
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

def objective_function_ode(model, args, method="finite"):

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
    residual = dx_dt + 5/9 * (-x + x*u + u**2)
    physics_loss = residual.pow(2).mean()

    # 4) end‐point control cost (e.g. minimize x(T)^2)
    control_cost = -x[:, -1].mean()

    # 6) total objective
    J = w[0] * control_cost \
      + w[1] * physics_loss \

    return J


def optimize_neural_operator(model, objective_function, m, end_time, num_epochs, learning_rate, bounds=[0,5], w=[1,1,1], model_name="lno"):  

    if model_name=="lno":
        u = (0.2 * torch.randn(1, m, 1, device=device))
        t = torch.tensor(torch.linspace(0, 1, m), dtype=torch.float).reshape(1, m, 1).to(device).repeat([u.shape[0], 1, 1]).clone().detach().requires_grad_(True)
    elif model_name=="fno":
        u = (0.2 * torch.randn(1, m, device=device))  # shape: [1, m]
        t = torch.linspace(0, end_time, m, device=device).unsqueeze(0).unsqueeze(-1)  # shape: [1, m, 1]

    u.requires_grad_(True)
    optimizer = torch.optim.Adam([u], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    for epoch in range(num_epochs):

        optimizer.zero_grad()

        if bounds is not None:
            u_min, u_max = bounds[0], bounds[1]
            u_bounded = u_min + (u_max - u_min) * 0.5 * (1 + torch.tanh(u))
        else:
            u_bounded = u
        
        x = model(u_bounded, t)

        args = {'u': u_bounded, 'x': x, 't': t, 'w':w}

        loss = objective_function(args)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, time: {datetime.now().time()}')

        scheduler.step()
    
    u = u.squeeze(-1)            # shape: [1, m]
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