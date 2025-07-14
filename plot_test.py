from utils.data import DiskBackedODEDataset
from utils.plotter import plot_validation_samples
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
from datetime import datetime
import os
from models.deeponet import DeepONetCartesianProd
import torch.optim as optim
import heapq

SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def gradient(x, t, batch_size, n_points, dim_x):
     # Physics loss
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


def calculate_test_errors(model: torch.nn.Module,
                          test_loader: torch.utils.data.DataLoader,
                          top_k: int = 20,
                          device: torch.device = None) -> dict:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    
    # Pre-compute t_uniform once since trajectory shape is consistent
    # Get trajectory length from dataset directly
    t_uniform = torch.linspace(0, 1, test_loader.dataset.m, device=device)
    
    # Initialize running statistics
    all_errors = []
    running_sum = 0.0
    running_sum_sq = 0.0
    total_samples = 0
    
    # For top-k tracking, use a min-heap to keep only the worst errors
    top_k_heap = []  # (error, sample_idx, u, trajectory, prediction)
    sample_idx = 0
    
    with torch.no_grad():
        for batch in test_loader:
            u, t, t0, ut, trajectory, mask = batch
            
            # Move batch to device if needed
            u = u.to(device)
            trajectory = trajectory.to(device)
            
            prediction = model(u, t_uniform)
            
            errors = relative_error(prediction, trajectory,
                                    dim=list(range(1, trajectory.ndim)))
            
            # Update running statistics
            batch_errors = errors.cpu()
            all_errors.append(batch_errors)
            running_sum += batch_errors.sum().item()
            running_sum_sq += (batch_errors ** 2).sum().item()
            total_samples += batch_errors.numel()
            
            # Update top-k heap
            for i in range(batch_errors.size(0)):
                error_val = batch_errors[i].item()
                
                if len(top_k_heap) < top_k:
                    # Add to heap if we haven't filled it yet
                    heapq.heappush(top_k_heap, (error_val, sample_idx + i, 
                                               u[i].cpu(), trajectory[i].cpu(), prediction[i].cpu()))
                elif error_val > top_k_heap[0][0]:
                    # Replace smallest error in heap
                    heapq.heapreplace(top_k_heap, (error_val, sample_idx + i,
                                                  u[i].cpu(), trajectory[i].cpu(), prediction[i].cpu()))
            
            sample_idx += batch_errors.size(0)
    
    # Concatenate all errors for final statistics
    errors_tensor = torch.cat(all_errors)
    mean_error = torch.mean(errors_tensor)
    std_error = torch.std(errors_tensor)
    
    # Extract top-k results from heap (sorted in ascending order, so reverse for worst errors)
    top_k_heap.sort(reverse=True)  # Sort by error value descending
    
    topk_errors = torch.tensor([item[0] for item in top_k_heap])
    topk_indices = torch.tensor([item[1] for item in top_k_heap])
    topk_u = [item[2] for item in top_k_heap]
    topk_trajectory = [item[3] for item in top_k_heap]
    topk_prediction = [item[4] for item in top_k_heap]

    return {
        'mean_error': mean_error,
        'std_error': std_error,
        'topk_indices': topk_indices,
        'topk_errors': topk_errors,
        'topk_u': topk_u,
        'topk_trajectory': topk_trajectory,
        'topk_prediction': topk_prediction
    }

            

def training(model, optimizer, train_loader, test_loader, num_epochs=1000, problem="linear", w=[1, 1, 1], semisupervised=False):

    checkpoint_path = f'trained_models/{problem}/deeponet/'
    plots_folder = f'plots/{problem}/deeponet/'

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    losses = []

    loss_function = compute_loss[problem]
    
    for epoch in range(num_epochs):

        model.train()
        batch_true_losses = []
        batch_physics_losses = []
        batch_initial_losses = []

        for u, t, t0, ut, trajectory, mask in train_loader:

            t.requires_grad_(True)

            x = model(u, t)
            x0 = model(u, t0)
            xf = 0

            batch_size = u.shape[0]
            n_points = t.shape[0]
            dim_x = x.shape[1]

            dx_dt = gradient(x, t, batch_size, n_points, dim_x)

            physics_loss = loss_function['physics_loss']({'x': x, 'dx_dt': dx_dt, 'u': ut})
            initial_loss = loss_function['initial_loss']({'x': x0})
            boundary_loss = loss_function['boundary_loss']({'x': xf})

            batch_physics_losses.append(physics_loss.item())
            batch_initial_losses.append(initial_loss.item())
            
            # Total loss
            loss = w[0]*physics_loss + w[1]*initial_loss + w[2]*boundary_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if semisupervised:
                supervised_mask = mask.bool()
                if supervised_mask.any():

                    t_uniform = torch.linspace(0, 1, 200, device=t.device).unsqueeze(1)
                    prediction = model(u[supervised_mask], t_uniform)
                    relative_error_train = relative_error(prediction, trajectory[supervised_mask])
                    true_loss = relative_error_train.item()
                    batch_true_losses.append(true_loss)
            
            print(f'True loss: {true_loss:.4f}, Physics loss: {physics_loss.item():.4f}, Initial loss: {initial_loss.item():.4f}')

        # Calculate test errors and log statistics
        test_errors = calculate_test_errors(model, test_loader, top_k=12, device=u.device)
        mean_error = test_errors['mean_error'].item()
        std_error = test_errors['std_error'].item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, '
              f'True loss: {np.mean(batch_true_losses):.4f}, Physics loss: {np.mean(batch_physics_losses):.4f}, Initial loss: {np.mean(batch_initial_losses):.4f}, '
              f'Test Error: {mean_error:.4f} ± {std_error:.4f}, '
              f'time: {datetime.now().time()}')

        # Plot top 12 worst predictions every 100 epochs
        if (epoch + 1) % 10 == 0:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            plot_name = f'epoch_{epoch+1}_worst_predictions_{timestamp}.png'
            
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

        # save model every 100th
        if (epoch + 1) % 100 == 0:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            model_filename = f'epoch[{epoch+1}]_model_time_[{timestamp}]_loss_[{loss.item():.4f}].pth'

            # save optimizer
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1
            }
            torch.save(checkpoint, f"{checkpoint_path}/{model_filename}")

    return model, losses


compute_loss = {
    'linear': {
        'physics_loss': lambda args: torch.mean((args['dx_dt'] + args['x'] - args['u'])**2),
        'initial_loss': lambda args: torch.mean((torch.ones_like(args['x']) - args['x'])**2),
        'boundary_loss': lambda args: 0,
    }
}

problems = ['linear', 'oscillatory', 'polynomial_tracking', 'nonlinear', 'singular_arc']
idx = 0

architecture = 'deeponet'
train_path = 'datasets/linear/[Train]-seed-1234-date-2025-07-14.pt'
test_path = 'datasets/linear/[Test]-seed-42-date-2025-07-14.pt'

train_ds = DiskBackedODEDataset(train_path, architecture=architecture)
test_ds = DiskBackedODEDataset(test_path, architecture=architecture)

# 2) Recreate your DataLoader exactly as before
train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
    collate_fn=train_ds.get_collate_fn()
)

test_loader = DataLoader(
    test_ds,
    batch_size=128,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
    collate_fn=test_ds.get_collate_fn()
)
folder = f'plots/{problems[idx]}/{architecture}'


# Model Parameters
m = 200         # sensor size (branch input size)
n_hid = 200     # layer's hidden sizes
p = 200         # output size
dim_x = 1       # trunk (trunk input size)
lr = 0.0001
num_epochs = 1000

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify the MLP architecture
branch_net = [m, n_hid, n_hid, n_hid, n_hid, p]
branch_activations = ['tanh', 'tanh','tanh', 'tanh', 'none']
trunk_net = [dim_x, n_hid, n_hid, n_hid, n_hid,  p]
trunk_activations = ['tanh', 'tanh', 'tanh', 'tanh', 'none']

# Initialize model
model = DeepONetCartesianProd(branch_net, trunk_net, branch_activations, trunk_activations)
model.to(device)

#Initialize Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

training(model, optimizer, train_loader, test_loader, num_epochs=num_epochs, problem=problems[idx], w=[1, 1, 1], semisupervised=True)
