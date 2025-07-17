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

def load_data(architecture, train_path, test_path, seed, batch_size=128):
    
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

def solve_optimization(
        model, problem, lr=0.001, architecture="deeponet",
        w=[100, 1], bounds=None, num_epochs=1000, m=200, device=None, 
        plot_interval=200):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if architecture == "deeponet":
        gradient = gradient_automatic
    else:
        gradient = gradient_finite_difference

    t_np = np.linspace(0, 1, m)
    t = torch.tensor(t_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1, m]

    u = torch.zeros((1, m), dtype=torch.float32, device=device, requires_grad=True)
    u_param = torch.nn.Parameter(u)
    optimizer = optim.Adam([u_param], lr=lr)

    objective_func = objective_functions[problem]
    physics_loss = compute_loss_uniform_grid[problem]['physics']
    optimal_x_fn = optimal_solutions[problem]['x']
    optimal_u_fn = optimal_solutions[problem]['u']

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        u_clamped = u_param
        if bounds:
            u_min, u_max = bounds
            u_clamped = u_param.clamp(u_min, u_max)

        if architecture == "deeponet":
            t_input = t.transpose(1, 0) if t.shape[0] == 1 else t   # DeepONet trunk shape
            x = model(u_clamped, t_input)
        else:
            x = model(u_clamped, t)
        dx_dt = gradient(x, t)

        loss = (
            w[0] * physics_loss({'x': x, 'dx_dt': dx_dt, 'u': u_clamped, 't': t})
            + w[1] * objective_func({'x': x, 'u': u_clamped, 't': t})
        )
        loss.backward()
        optimizer.step()

        if bounds:
            with torch.no_grad():
                u_param.data.clamp_(u_min, u_max)

        # --- PLOT INSIDE LOOP ---
        # Plot every 'plot_interval' epochs and at the final epoch
        if ((epoch + 1) % plot_interval == 0) or (epoch + 1 == num_epochs):
            print(f"Plotting at epoch {epoch+1}")
            plot_optimal_vs_predicted(
                t, u_param.detach(), x.detach(), optimal_u_fn, optimal_x_fn,
                title=f'Optimal vs Predicted (Epoch {epoch+1})'
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