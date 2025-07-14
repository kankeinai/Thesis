import torch
import heapq
from utils.plotter import plot_validation_samples, plot_analytics
import os
from datetime import datetime
import numpy as np


def gradient_deep_o_net(x, t, batch_size, n_points, dim_x):
     # Physics loss
    dx = torch.zeros(batch_size, n_points, dim_x, device=t.device)
    # This loop is a bottleneck but i havent found a way to parallize this efficiently
    for b in range(batch_size):
        # Compute gradients for each batch independently
        dx[b] = torch.autograd.grad(x[b], t, torch.ones_like(x[b]), create_graph=True)[0]

    dx_dt = dx[:,:,0]

    return dx_dt

def gradient_deep_o_net(x, t):
     # Physics loss
    batch_size = x.shape[0]
    n_points = t.shape[0]
    dim_x = x.shape[1]

    dx = torch.zeros(batch_size, n_points, dim_x, device=t.device)
    # This loop is a bottleneck but i havent found a way to parallize this efficiently
    for b in range(batch_size):
        # Compute gradients for each batch independently
        dx[b] = torch.autograd.grad(x[b], t, torch.ones_like(x[b]), create_graph=True)[0]

    dx_dt = dx[:,:,0]

    return dx_dt

def relative_error(prediction: torch.Tensor,
                   trajectory: torch.Tensor,
                   dim: int = None,
                   eps: float = 1e-8) -> torch.Tensor:
    
    num = torch.sqrt(torch.sum((prediction - trajectory) ** 2, dim=dim))
    den = torch.sqrt(torch.sum(trajectory ** 2, dim=dim)).clamp(min=eps)
    return num / den


import heapq
import torch

def calculate_test_errors(model: torch.nn.Module,
                          test_loader: torch.utils.data.DataLoader,
                          top_k: int = None,
                          device: torch.device = None, architecture='deeponet') -> dict:
    """
    Compute mean/std of your relative error over the test set;
    optionally also return the top‐k worst errors (with inputs/targets/preds).
    
    If top_k is None or <= 0, skips all top‐k bookkeeping.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    # Pre‐compute uniform time grid
    if architecture == 'deeponet':
        t_uniform = torch.linspace(0, 1, test_loader.dataset.m, device=device).unsqueeze(1)

    running_sum = 0.0
    running_sum_sq = 0.0
    total_samples = 0

    # only initialize heap if top_k requested
    if top_k and top_k > 0:
        top_k_heap = []  # will store (error, idx, u, traj, pred)
    sample_idx = 0
    all_errors = []

    with torch.no_grad():
        for batch in test_loader:
            if architecture == 'deeponet':
                u, t, t0, ut, trajectory, mask = batch
                t_uniform = t_uniform.to(device)
                pred = model(u, t_uniform)
            else:
                u, t,  trajectory, mask = batch
                pred = model(u, t)
            errors = relative_error(pred, trajectory,
                                    dim=list(range(1, trajectory.ndim)))

            batch_errors = errors.cpu()
            all_errors.append(batch_errors)
            running_sum += batch_errors.sum().item()
            running_sum_sq += (batch_errors ** 2).sum().item()
            total_samples += batch_errors.numel()

            # only do top‐k tracking if requested
            if top_k and top_k > 0:
                for i, err_val in enumerate(batch_errors):
                    err = err_val.item()
                    if len(top_k_heap) < top_k:
                        heapq.heappush(top_k_heap, (err, sample_idx + i,
                                                   u[i].cpu(), trajectory[i].cpu(), pred[i].cpu()))
                    elif err > top_k_heap[0][0]:
                        heapq.heapreplace(top_k_heap, (err, sample_idx + i,
                                                       u[i].cpu(), trajectory[i].cpu(), pred[i].cpu()))
            sample_idx += batch_errors.size(0)

    errors_tensor = torch.cat(all_errors)
    mean_error = errors_tensor.mean()
    std_error = errors_tensor.std()

    result = {
        'mean_error': mean_error,
        'std_error': std_error,
    }
    

    # unpack top‐k only if we did the work
    if top_k and top_k > 0:
        # reverse‐sort so worst first
        top_k_heap.sort(reverse=True)
        result.update({
            'topk_indices': torch.tensor([item[1] for item in top_k_heap]),
            'topk_errors':  torch.tensor([item[0] for item in top_k_heap]),
            'topk_u':       [item[2] for item in top_k_heap],
            'topk_trajectory':   [item[3] for item in top_k_heap],
            'topk_prediction':   [item[4] for item in top_k_heap],
        })

    return result


def training(model, optimizer, scheduler, train_loader, test_loader, compute_loss, gradient, num_epochs=1000, architecture="deeponet", problem="linear", w=[1, 1], save = 10):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f'trained_models/{problem}/{architecture}/attempt_started{timestamp}'
    plots_folder = f'plots/{problem}/{architecture}/{timestamp}'

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    losses = {
        'train_loss': [],
        'physics_loss': [],
        'initial_loss': [],
        'test_loss': []
    }
    
    for epoch in range(num_epochs):

        model.train()
        batch_true_losses = []
        batch_physics_losses = []
        batch_initial_losses = []

        for batch in train_loader:

            if architecture == 'deeponet':
                u, t, t0, ut, trajectory, mask = batch
                t.requires_grad_(True)
                t_uniform = torch.linspace(0, 1, u.shape[1], device=u.device).unsqueeze(1)
            else:
                u, t, trajectory, mask = batch

            optimizer.zero_grad()

            x = model(u, t)

            if architecture == 'deeponet':
                x0 = model(u, t0)
            else:
                x0 = x[:, 0]

            dx_dt = gradient(x, t)

            physics_loss = compute_loss['physics_loss']({'x': x, 'dx_dt': dx_dt, 'u': ut if architecture == 'deeponet' else u})
            initial_loss = compute_loss['initial_loss']({'x': x0})

            batch_physics_losses.append(physics_loss.item())
            batch_initial_losses.append(initial_loss.item())
            
            # Total loss
            loss = w[0]*physics_loss + w[1]*initial_loss

            
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                labels_mask = mask.bool()
                if labels_mask.any():
                    prediction = model(u[labels_mask], t_uniform if architecture=="deeponet" else t[labels_mask])
                    relative_error_train = relative_error(prediction, trajectory[labels_mask])
                    true_loss = relative_error_train.item()
                    batch_true_losses.append(true_loss)
        
        last_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        scheduler.step()

        if (epoch+1) % save == 0:
            test_errors = calculate_test_errors(model, test_loader, top_k=12, device=u.device, architecture=architecture)
            plot_name = f'epoch_{epoch+1}_test_predictions.png'
            
            # Create worst_data dictionary
            worst_data = {
                'trajectory': test_errors['topk_trajectory'],
                'u': test_errors['topk_u'],
                'predicted': test_errors['topk_prediction'],
                'errors': test_errors['topk_errors'].tolist(),
                'indices': test_errors['topk_indices'].tolist(),
            }
            
            # Use existing plot_validation_samples function
            plot_validation_samples(
                data=worst_data,
                epoch=epoch + 1,
                folder=plots_folder,
                name=plot_name,
                grid_rows=4,
                grid_cols=3,
            )
        else:
            test_errors = calculate_test_errors(model, test_loader, top_k=None, device=u.device, architecture=architecture)

        mean_error = test_errors['mean_error']
        std_error = test_errors['std_error']
        physics_loss = np.mean(batch_physics_losses)
        initial_loss = np.mean(batch_initial_losses)
        true_loss = np.mean(batch_true_losses) 

        old_lrs = [g['lr'] for g in optimizer.param_groups]
        scheduler.step(physics_loss)
        new_lrs = [g['lr'] for g in optimizer.param_groups]

        if new_lrs != old_lrs:
            print(f"Epoch {epoch:03d}: reducing LR → {new_lrs}")

        losses['train_loss'].append(true_loss)
        losses['physics_loss'].append(physics_loss)
        losses['initial_loss'].append(initial_loss)
        losses['test_loss'].append(mean_error.item())


        if (epoch+1) % save == 0:
            plot_analytics(losses, epoch+1, timestamp, last_timestamp, problem=problem, architecture=architecture)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Physics loss: {physics_loss:.4f}, Initial loss: {initial_loss:.4f}, '
              f'Train loss: {true_loss:.4f}, Test loss: {mean_error:.4f} ± {std_error:.4f}, '
              f'time: {datetime.now().time()}')


        # save model every 10th
        if (epoch + 1) % save == 0:
            model_filename = f'epoch[{epoch+1}]_model_time_[{last_timestamp}]_loss_[{loss.item():.4f}].pth'

            # save optimizer
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1
            }
            torch.save(checkpoint, f"{checkpoint_path}/{model_filename}")

    return model, losses
