
import torch
import random
import numpy as np
from models.lno import LNO1d, LNO1d_extended
import torch.optim as optim
from utils.training import training
from utils.scripts import load_data
from torch.optim.lr_scheduler import StepLR
from utils.settings import compute_loss_uniform_grid, datasets, gradient_finite_difference

SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
            
problems = ['linear', 'oscillatory', 'polynomial_tracking', 'nonlinear', 'singular_arc']
idx = 3
compute_loss = compute_loss_uniform_grid[problems[idx]]

architecture = 'lno'

train_loader, test_loader = load_data(
    architecture,
    datasets[problems[idx]]['train'],
    datasets[problems[idx]]['validation'],
    seed = SEED,
)

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================================
# Model definition
# ====================================
modes = 32
width = 16
hidden_layer = 128

model = LNO1d(width, modes, activation = "silu",batch_norm=True, active_last=True, hidden_layer=hidden_layer).to(device)
# ====================================
# Training settings
# ====================================
#Initialize Optimizer
lr = 0.001
print(f"Using learning rate: {lr}")
epochs = 10000

optimizer = optim.Adam(model.parameters(), lr=lr)

scheduler = StepLR(
    optimizer,
    step_size=100,   # decay LR every 50 epochs (set as you like)
    gamma=0.9,      # halve the LR
)


training(model, optimizer, scheduler, train_loader, test_loader, compute_loss, gradient_finite_difference, architecture=architecture, num_epochs=epochs, finetuning = False, save = 10, save_plot = 10, early_stopping_patience=100, problem=problems[idx], w=[1, 1, 0.01])
