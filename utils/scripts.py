from utils.data import DiskBackedODEDataset
import torch
from torch.utils.data import DataLoader
from utils.settings import gradient_automatic, gradient_finite_difference
import numpy as np
import torch.optim as optim
from utils.settings import objective_functions, compute_loss_uniform_grid, optimal_solutions
from utils.plotter import plot_optimal_vs_predicted
from utils.metrics import calculate_true_error
import os
import numpy as np
from scipy.interpolate import interp1d
from utils.data import ode_registry

def load_data(architecture, train_path, test_path, seed, batch_size=64):
    
    train_ds = DiskBackedODEDataset(train_path, architecture=architecture)
    test_ds = DiskBackedODEDataset(test_path, architecture=architecture)

    # 2) Recreate your DataLoader exactly as before
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=train_ds.get_collate_fn()
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=test_ds.get_collate_fn()
    )

    return train_loader, test_loader

def smoothness_penalty(u):
    # u: (batch, N)
    return torch.mean((u[:, 1:] - u[:, :-1])**2)

def solve_optimization(
        model, problem, lr=0.001, architecture="deeponet",
        w=[100, 1], bounds=None, num_epochs=1000, m=200, device=None, 
        plot_interval=200):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    t_np = np.linspace(0, 1, m)

    if architecture == "deeponet":
        t = torch.tensor(t_np, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
        gradient = gradient_automatic
    else:
        t = torch.tensor(t_np, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)
        gradient = gradient_finite_difference

    u = torch.ones((1, m), dtype=torch.float32, device=device, requires_grad=True)*1.5
    u_param = torch.nn.Parameter(u)
    optimizer = optim.Adam([u_param], lr=lr)

    problem_instance = ode_registry.get(problem)

    objective_func = objective_functions[problem]
    physics_loss = compute_loss_uniform_grid[problem]['physics_loss']
    initial_loss = compute_loss_uniform_grid[problem]['initial_loss']
    boundary_loss = compute_loss_uniform_grid[problem]['boundary_loss']


    optimal_x_fn = optimal_solutions[problem]['x']({'t': t_np})
    optimal_u_fn = optimal_solutions[problem]['u']({'t': t_np})
    optimal_objective_value = objective_func({'u': torch.tensor(optimal_u_fn).to(device), 'x': torch.tensor(optimal_x_fn).to(device), 't': t})

    x_pred_2 = compute_predicted_trajectory(problem_instance, optimal_u_fn, t_np.reshape(200))

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        u_clamped = u_param
        if bounds:
            u_min, u_max = bounds
            u_clamped = u_param.clamp(u_min, u_max)

    
        x = model(u_clamped, t)
        dx_dt = gradient(x, t)


        loss = (
            w[0] * physics_loss({'x': x, 'dx_dt': dx_dt, 'u': u_clamped, 't': t})
            + w[1] * objective_func({'x': x.squeeze(), 'u': u_clamped.squeeze(), 't': t.squeeze()}) + w[2] * initial_loss({'x': x.view(200)[0]})+
            w[3]* smoothness_penalty(u_clamped) + w[4]*boundary_loss({'x':x[-1]})
        )
        loss.backward()
        optimizer.step()

        if bounds:
            with torch.no_grad():
                u_param.data.clamp_(u_min, u_max)

        # --- PLOT INSIDE LOOP ---
        # Plot every 'plot_interval' epochs and at the final epoch
        if ((epoch + 1) % plot_interval == 0) or (epoch + 1 == num_epochs):

            optimal_objective_value_predicted = objective_func({'u': u_param.squeeze(), 'x': x.squeeze(), 't': t.squeeze()})
            title = f"predicted value: {optimal_objective_value_predicted}, correct value: {optimal_objective_value}"

            u_pred = u_param.detach().cpu().numpy().reshape(200)

            # Step 3: Compute x_pred
            x_pred = compute_predicted_trajectory(problem_instance, u_pred, t_np.reshape(200))


            print(f"Plotting at epoch {epoch+1}")
            plot_optimal_vs_predicted(
                t_np.reshape(200), u_pred, x_pred, x.detach(), optimal_u_fn, optimal_x_fn, x_pred_2,
                title=title, savepath=f"found_trajectories/{problem}/{architecture}/plot.png"
            )

        # Calculate relative error for monitoring
        rel_err_u, rel_err_x = calculate_true_error(
            x.detach(), u_param.detach(), t, optimal_x_fn, optimal_u_fn, device
        )
        print(f"Epoch {epoch+1:4d} | Loss: {loss.item():.6f} | rel_err_u: {rel_err_u:.4f}, rel_err_x: {rel_err_x:.4f}")

def save_model(model, optimizer, epoch, timestamp, mean_error, checkpoint_path, model_filename = None):
    # save model every 10th
    if model_filename is None:
        model_filename = f'epoch[{epoch+1}]_model_time_[{timestamp}]_loss_[{mean_error.item():.4f}].pth'
    # save optimizer
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch + 1
    }
    torch.save(checkpoint, f"{checkpoint_path}/{model_filename}")

def check_that_folder_exists(folders):
    for f in folders:
        os.makedirs(f, exist_ok=True)

def compute_predicted_trajectory(problem, u_pred: np.ndarray, t_grid: np.ndarray):
    """
    Compute x(t) by solving the ODE with a predicted control u_pred(t),
    defined on a uniform time grid t_grid.

    Args:
        problem: An instance of ODEProblem (e.g. from registry)
        u_pred: 1D numpy array, predicted control values (shape: [m])
        t_grid: 1D numpy array, time discretization grid (shape: [m])

    Returns:
        x_pred: 1D numpy array, predicted state trajectory (shape: [m])
    """
    # Step 1: Interpolate the predicted control function
    control_func = interp1d(t_grid, u_pred, kind='cubic', fill_value="extrapolate")

    # Step 2: Use ODEProblem to solve the resulting trajectory
    x_pred = problem.solve_trajectory(control_func=control_func, t_eval=t_grid)

    return x_pred