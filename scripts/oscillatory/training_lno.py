from utils.data import DiskBackedODEDataset
from utils.plotter import plot_validation_samples
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
from datetime import datetime
import os
from models.lno import LNO1d
import torch.optim as optim
import heapq
from utils.training import gradient_deep_o_net,  training
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
            


compute_loss = {
    'linear': {
        'physics_loss': lambda args: torch.mean((args['dx_dt'] + args['x'] - args['u'])**2),
        'initial_loss': lambda args: (args['x'-1] - 1.0).pow(2).mean()
    },
    'oscillatory': {
        'physics_loss': lambda args: torch.mean((args['dx_dt']  - args['u'] - torch.cos(4*torch.pi*args['t']))**2),
        'initial_loss': lambda args: (args['x'] - 0).pow(2).mean(),
    },
    'polynomial_tracking': {
        'physics_loss': lambda args: torch.mean((args['dx_dt'] - args['u'])**2),
        'initial_loss': lambda args: (args['x'] - 0).pow(2).mean(),
    },
    'nonlinear': {
        'physics_loss': lambda args: torch.mean((args['dx_dt'] - 5/2*( - args['x'] + args['x']* args['u'] - args['u']**2))**2),
        'initial_loss': lambda args: (args['x'] - 1).pow(2).mean(),
    },
    'singular_arc': {
        'physics_loss': lambda args: torch.mean((args['dx_dt'] - (args['x']**2 +  args['u']))**2),
        'initial_loss': lambda args: (args['x'] - 1).pow(2).mean(),
    }
}

problems = ['linear', 'oscillatory', 'polynomial_tracking', 'nonlinear', 'singular_arc']
idx = 1

architecture = 'lno'
train_path = f'datasets/{problems[idx]}/[Train]-seed-1234-date-2025-07-15.pt'
test_path = f'datasets/{problems[idx]}/[Test]-seed-42-date-2025-07-15.pt'

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
    collate_fn=test_ds.get_collate_fn(),
)


# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================================
# Model definition
# ====================================
modes = 16
width = 4
model = LNO1d(width,modes, hidden_layer=32).to(device)

# ====================================
# Training settings
# ====================================
step_size = 1
gamma = 0.5
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
stepsize= 200
scheduler = StepLR(optimizer, step_size=stepsize, gamma=0.5)
num_epochs = 1000

def gradient_lno(x, t):
    dx_dt,  = torch.gradient(x, spacing=(t[0, :],), dim=1)
    return dx_dt.squeeze(-1)

training(model, optimizer, scheduler, train_loader, test_loader, compute_loss[problems[idx]], gradient_lno, architecture='lno', num_epochs=num_epochs, problem=problems[idx], w=[1, 1])
