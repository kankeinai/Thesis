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
from utils.training import gradient_automatic, training, load_data
from torch.optim.lr_scheduler import StepLR
from utils.settings import compute_loss_random_grid

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
compute_loss = compute_loss_random_grid[problems[idx]]

architecture = 'deeponet'

train_loader, test_loader = load_data(
    problems[idx],
    architecture,
    '[Train]-seed-1234-date-2025-07-15.pt',
    '[Test]-seed-42-date-2025-07-15.pt',
    SEED,
)

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Parameters
m = 200         # sensor size (branch input size)
n_hid = 250     # layer's hidden sizes
p = 200         # output size
dim_x = 1       # trunk (trunk input size)

# Specify the MLP architecture
branch_net = [m, n_hid,  n_hid, n_hid, n_hid, p]
branch_activations = ['tanh', 'tanh', 'tanh', 'tanh','tanh','none']
trunk_net = [dim_x, n_hid,  n_hid, n_hid, n_hid, p]
trunk_activations = ['tanh', 'tanh', 'tanh', 'tanh','tanh','none']
model = DeepONetCartesianProd(branch_net, trunk_net, branch_activations, trunk_activations)
model.to(device)

#Initialize Optimizer
lr = 0.0001
stepsize = 200
epochs = 1000

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=stepsize, gamma=0.5)


training(model, optimizer, scheduler, train_loader, test_loader, compute_loss, gradient_automatic, num_epochs=epochs, problem=problems[idx], w=[1, 1])
