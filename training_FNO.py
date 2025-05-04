import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.data import MultiFunctionDatasetODE, custom_collate_ODE_fn_fno
from models.fno import *
from utils.scripts import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# Dataset generation
# -------------------------
n_functions = 1000000
end_time = 1.0
m = 200
batch_size = 1024
t_grid = torch.linspace(0, end_time, m).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1).to(device)
t_grid.requires_grad = True

print("===============================\nStarted generating dataset")

dataset = MultiFunctionDatasetODE(
    m=m,
    n_functions=n_functions,
    function_types=['grf', 'linear', 'sine', 'polynomial', 'constant'],
    end_time=end_time,
    num_domain=m,
    num_initial=20,
    grf_lb=0.02,
    grf_ub=0.5
)

dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_ODE_fn_fno, shuffle=True)
print("===============================\nDataset is ready")

# -------------------------
# Model Definition
# -------------------------
model = FNO1d(modes=32, width=64).to(device)

step_size = 3
gamma = 0.9
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# -------------------------
# Train
# -------------------------
epochs = 10
print("===============================\nTraining started")
trained_model = train_fno(model, compute_loss_nde, dataloader, optimizer, scheduler, epochs, t_grid, logging=True)
