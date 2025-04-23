import torch
from datetime import datetime
import os
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_loss(model, u, t, model_name="fno"): 
    
    """
    Compute the combined physics-informed and initial-condition loss for a neural operator.

    Parameters
    ----------
    model : nn.Module
        The trained neural operator (e.g., FNO1d or LNO).
    u : torch.Tensor, shape (B, N)
        Input field values at each spatial location for each batch example.
    t : torch.Tensor, shape depends on model
        Coordinate or time grid corresponding to u. For FNO, shape (B, N, 1).
    model_name : str, optional
        Identifier for model type; affects finite-difference spacing:
        - "fno": uses t[0, 1] - t[0, 0]
        - "lno": uses t[1] - t[0]

    Returns
    -------
    physics_loss : torch.Tensor
        Mean squared residual of the PDE approximation.
    initial_loss : torch.Tensor
        Mean squared error at the initial spatial point (boundary) against 1.0.
    """
    x = model(u, t)
    if model_name=="lno":
        dt=(t[1]-t[0]).item()
    elif model_name=="fno":
        dt=(t[0,1]-t[0, 0]).item()
    dx_dt = (x[:, 1:, :] - x[:, :-1, :]) / dt  # crude finite diff

    if model_name=="lno":
        residual = dx_dt + x[:, :-1, :] - u[:, :-1, :]
    elif model_name=="fno":
        residual = dx_dt + x[:, :-1, :] - u[:, :-1].unsqueeze(-1)

    physics_loss = torch.mean(residual ** 2)
    initial_loss = torch.mean((x[:, 0, :] - 1.0)**2)
    return physics_loss, initial_loss


def train_fno(model, dataloader, optimizer, scheduler, epochs, t_grid, w=[1,1], folder = "trained_models/fno", logging = True):
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
            t = t_grid[:u.shape[0], :].to(device)
            optimizer.zero_grad()
            
            physics_loss, initial_loss = compute_loss(model, u, t, model_name="fno")
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

def train_lno(model, dataloader, optimizer, scheduler, epochs, w = [1,1], folder = "trained_models/lno", logging = True):
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

        for u, t, _, _ in dataloader:

            u = u.to(device).unsqueeze(-1)
            optimizer.zero_grad()

            physics_loss, initial_loss = compute_loss(model, u, t, model_name="lno")
            loss = w[0] * physics_loss + w[1] * initial_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if logging:
                print(f"Epoch: {epoch}, Physics loss: {physics_loss}, Initial loss: {initial_loss}")

        epoch_loss /= n_batches

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, Time: {(datetime.now() - start_time).total_seconds()} s') 

        if (epoch + 1) % 10 == 0:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            os.makedirs(folder, exist_ok=True)
            model_filename = f'epochs_[{epoch+1}]_model_time_[{timestamp}]_loss_[{epoch_loss:.4f}].pth'
            torch.save(model.state_dict(), folder+f"/{model_filename}")

        scheduler.step()

def objective_function(args, model_name):

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

    x = args['x']
    u = args['u']
    t = args['t']
    w = args['w']

    if model_name=="lno":
        dt = (t[1] - t[0]).item()
    elif model_name=="fno":
        dt = (t[0, 1] - t[0, 0]).item()


    dx_dt = (x[:, 1:, :] - x[:, :-1, :]) / dt  # crude finite diff

    if model_name=="lno":
        residual = dx_dt + x[:, :-1, :] - u[:, :-1, :]
    elif model_name=="fno":
        residual = dx_dt + x[:, :-1, :] - u[:, :-1].unsqueeze(-1)

    physics_loss = torch.mean(residual ** 2)
    initial_loss = torch.mean((x[:, 0] - torch.ones(10, device = device))**2)
    control_cost = torch.mean(torch.trapz((x**2 + u**2).squeeze(), t.squeeze()))

    J = w[0] * control_cost + w[1] * physics_loss + w[2] * initial_loss 
    return J

def optimize_neural_operator(model, objective_function, m, end_time, num_epochs, learning_rate, w=[1,1,1], model_name="lno"):  

    """
    Optimize the control signal u to minimize the objective via gradient descent.

    Parameters
    ----------
    model : nn.Module
        Pretrained neural operator mapping (u, t) → x.
    objective_function : callable
        Function that computes J(args, model_name).
    m : int
        Number of time discretization points.
    end_time : float
        End of time interval [0, end_time].
    num_epochs : int
        Number of optimization steps.
    learning_rate : float
        Learning rate for the control optimizer.
    w : list of float, optional
        Weights for [control_cost, physics_loss, initial_loss].
    model_name : str, optional
        'lno' or 'fno' to match objective implementation.

    Returns
    -------
    u : torch.Tensor
        Optimized control signal, shape depends on model.
    x : torch.Tensor
        Final state trajectory from model(u, t).
    t : torch.Tensor
        Time grid used in optimization.
    """
     
    if model_name=="lno":
        u = (0.2 * torch.randn(1, m, device=device)).unsqueeze(-1)  # shape = (1, m, 1)
        t = torch.linspace(0, end_time, m, device= device)
    elif model_name=="fno":
        u = (0.2 * torch.randn(1, m, device=device))  # shape: [1, m]
        t = torch.linspace(0, end_time, m, device=device).unsqueeze(0).unsqueeze(-1)  # shape: [1, m, 1]

    u.requires_grad_(True)
    optimizer = torch.optim.Adam([u], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    for epoch in range(num_epochs):

        optimizer.zero_grad()
        x = model(u, t)

        args = {'u': u, 't': t, 'x':x, 'w':w}
        loss = objective_function(args, model_name)
        loss.backward()
        optimizer.step()


        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, time: {datetime.now().time()}')

        scheduler.step()

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