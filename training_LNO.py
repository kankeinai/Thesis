import torch
from utils.data import MultiFunctionDatasetODE, custom_collate_ODE_fn_fno 
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

dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_ODE_fn_fno, shuffle=True)
print("===============================\nDataset is ready")
# ====================================
# Model definition
# ====================================
modes = 32
width = 8
model = LNO1d(width,modes).cuda()

# ====================================
# Training settings
# ====================================
step_size = 100
gamma = 0.9
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


# ====================================
# Train
# ====================================
epochs = 100

print("===============================\nTraining started")
trained_model = train_lno(model, dataloader, optimizer, scheduler, epochs, logging=False)