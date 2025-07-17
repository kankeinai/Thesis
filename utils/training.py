import torch
from utils.plotter import plot_validation_samples, plot_analytics
import os
from datetime import datetime
import numpy as np
import time
import torch
from utils.metrics import calculate_test_errors, relative_error
from utils.scripts import save_model, check_that_folder_exists


def training(model, optimizer, scheduler, train_loader, test_loader, compute_loss, gradient, num_epochs=1000, architecture="deeponet", problem="linear", w=[1, 1], save_plot = 20, save=100, early_stopping_patience=100, min_epoch=0,  checkpoint_path=None, plots_folder=None, analytics_folder=None, time_folder=None):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if checkpoint_path is None:
        checkpoint_path = f'trained_models/{problem}/{architecture}/attempt_started{timestamp}/'
    if plots_folder is None:
        plots_folder = f'plots/{problem}/{architecture}/attempt_started{timestamp}/'
    if analytics_folder is None:
        analytics_folder = f'analytics_plots/{problem}/{architecture}/attempt_started{timestamp}/'
    if time_folder is None:
        time_folder = f'computation_time/{problem}/{architecture}/attempt_started{timestamp}/'

    best_model_path = f'trained_models/{problem}/{architecture}/attempt_started{timestamp}/best/'

    check_that_folder_exists([checkpoint_path, plots_folder, analytics_folder, time_folder, best_model_path])

    losses = {
        'train_loss': [],
        'physics_loss': [],
        'initial_loss': [],
        'test_loss': []
    }

    best_phys_loss = float('inf')
    epochs_since_improvement = 0
    epoch_times = []

    prev_lr = optimizer.param_groups[0]['lr']
    
    for epoch in range(min_epoch, min_epoch + num_epochs + 1):


        epoch_start = time.time()
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

            batch_size = u.shape[0]
            physics_loss = compute_loss['physics_loss']({'x': x, 'dx_dt': dx_dt, 'u': ut if architecture == 'deeponet' else u, 't': t.squeeze(-1).repeat(batch_size, 1) if architecture == 'deeponet' else t})
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

        core_end = time.time()
        epoch_time = core_end - epoch_start
        epoch_times.append(epoch_time)

        last_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

        if (epoch+1) % save_plot == 0:
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

        losses['train_loss'].append(true_loss)
        losses['physics_loss'].append(physics_loss)
        losses['initial_loss'].append(initial_loss)
        losses['test_loss'].append(mean_error.item())

        if (epoch+1) % save_plot == 0:
            plot_analytics(losses, epoch+1, last_timestamp, analytics_folder)

        print(f'Epoch [{epoch+1}/{min_epoch+num_epochs}], '
              f'Physics loss: {physics_loss:.4f}, Initial loss: {initial_loss:.4f}, '
              f'Train loss: {true_loss:.4f}, Test loss: {mean_error:.4f} ± {std_error:.4f}, '
              f'time: {datetime.now().time()}')
        
        if (epoch + 1) % save == 0:
            np.save(os.path.join(time_folder, f"time_epoch_{epoch+1}.npy"), np.array(epoch_times))
            print(f"Scheduled saving of model at epoch: {epoch + 1} with validation loss {mean_error.item():.4f}")
            save_model(model, optimizer, epoch, timestamp, mean_error, checkpoint_path)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if current_lr != prev_lr:
            print(f"Epoch {epoch+1}: learning rate changed to {current_lr:.6e}")
            prev_lr = current_lr

        if physics_loss.item() <  best_phys_loss - 1e-6:
            best_phys_loss = physics_loss.item()
            epochs_since_improvement = 0
            save_model(model, optimizer, epoch, timestamp, mean_error, best_model_path, model_filename =f"best_model.pt")
            print(f"Epoch {epoch+1}: saving model with validation loss {mean_error.item():.4f} physics loss {physics_loss.item()}")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement > early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model, losses