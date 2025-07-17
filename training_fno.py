
import torch
import random
import numpy as np
from models.fno import FNO1d
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
idx = 0
compute_loss = compute_loss_uniform_grid[problems[idx]]


architecture = 'fno'

train_loader, test_loader = load_data(
    problems[idx],
    architecture,
    datasets[problems[idx]]['train'],
    datasets[problems[idx]]['validation'],
    SEED,
)

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ====================================
# Model definition
# ====================================

modes = 32
width = 16
depth = 4
hidden_layer = 128

model = FNO1d(modes=modes, width=width, depth=depth, activation="silu", hidden_layer = hidden_layer).to(device)

# ====================================
# Training settings
# ====================================

#Initialize Optimizer
lr = 0.0001
print(f"Using learning rate: {lr}")
epochs = 1000

optimizer = optim.Adam(model.parameters(), lr=lr)

scheduler = StepLR(
    optimizer,
    step_size=100,   # decay LR every 50 epochs (set as you like)
    gamma=0.5,      # halve the LR
)

training(model, optimizer, scheduler, train_loader, test_loader, compute_loss, gradient_finite_difference, architecture=architecture, num_epochs=epochs, save = 10, save_plot = 10, problem=problems[idx], w=[1, 1])
