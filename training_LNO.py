import torch
from utils.data import MultiFunctionDatasetODE, custom_collate_ODE_fn
from torch.utils.data import DataLoader
from models.lno import *
from utils.scripts import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================================
# Dataset generation
# ====================================
n_functions = 1000000
grf_lb = 0.02
grf_ub = 0.5
end_time = 1.0
num_domain = 200
num_initial = 20
batch_size = 1024
m = 200 #number of sensors

print("===============================\nStarted generating dataset")

dataset = MultiFunctionDatasetODE(
    m=m,
    n_functions=n_functions,
    function_types=['grf', 'linear', 'sine', 'polynomial','constant'],
    end_time = end_time,
    num_domain = num_domain,
    num_initial = num_initial,
    grf_lb = grf_lb,
    grf_ub = grf_ub
)

t_grid  = torch.tensor(np.linspace(0, 1, m), dtype=torch.float).reshape(1, m, 1).to(device)

dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_ODE_fn, shuffle=True)
print("===============================\nDataset is ready")
# ====================================
# Model definition
# ====================================
modes = 32
width = 8
model = LNO1d(width,modes, hidden_layer=128).cuda()

# ====================================
# Training settings
# ====================================
step_size = 10
gamma = 0.9
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


# ====================================
# Train
# ====================================
epochs = 100

print("===============================\nTraining started")
trained_model = train_lno(model, compute_loss_nde, dataloader, optimizer, scheduler, epochs, t_grid, save=5, method="finite", logging=True)