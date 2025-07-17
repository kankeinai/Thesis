import torch
import random
import numpy as np
from models.lno import LNO1d
from utils.training import  training, load_data, gradient_finite_difference
from torch.optim.lr_scheduler import  StepLR
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
idx = 1
compute_loss = compute_loss_uniform_grid[problems[idx]]

architecture = 'lno'

train_loader, test_loader = load_data(
    problems[idx],
    architecture,
    SEED,
    batch_size=128,
    train_path=f'[Train]-seed-1234-date-2025-07-15.pt',
    test_path=f'[Test]-seed-42-date-2025-07-15.pt'
)

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================================
# Model definition
# ====================================
modes = 16
width = 4
model = LNO1d(width, modes, hidden_layer=32).to(device)

# ====================================
# Training settings
# ====================================
stepsize = 50
learning_rate = 0.0001
num_epochs = 1000

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=stepsize, gamma=0.5)


training(model, optimizer, scheduler, train_loader, test_loader, compute_loss, gradient_finite_difference, architecture='lno', num_epochs=num_epochs, problem=problems[idx], w=[1, 1])
