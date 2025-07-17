import torch
import random
import numpy as np
from models.fno import FNO1d
from utils.training import training, gradient_finite_difference, load_data
from torch.optim.lr_scheduler import StepLR
from utils.settings import compute_loss_uniform_grid

SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
            

problems = ['linear', 'oscillatory', 'polynomial_tracking', 'nonlinear', 'singular_arc']
idx = 2
compute_loss = compute_loss_uniform_grid[problems[idx]]

architecture = 'fno'

train_loader, test_loader = load_data(
    problems[idx],
    architecture=architecture,
    batch_size=128,
    seed=SEED,
    train_name=f'[Train]-seed-1234-date-2025-07-15.pt',
    test_name=f'[Test]-seed-42-date-2025-07-15.pt'
)

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ====================================
# Model definition
# ====================================


model = FNO1d(modes=32, width=16, depth=4, activation="silu", hidden_layer = 16).to(device)

# ====================================
# Training settings
# ====================================

learning_rate = 0.0001
stepsize= 50
num_epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=stepsize, gamma=0.5)

training(model, optimizer, scheduler, train_loader, test_loader, compute_loss, gradient_finite_difference, architecture=architecture, num_epochs=num_epochs, problem=problems[idx], w=[1, 1])
