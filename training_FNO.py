import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.data import MultiFunctionDatasetODE, custom_collate_ODE_fn_fno
from models.model_fno import *
from utils.scripts import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------
# Dataset generation
# -------------------------
n_functions = 1000000
end_time = 1.0
m = 200
batch_size = 1024
t_grid = torch.linspace(0, end_time, m).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
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

# -------------------------
# Model Definition
# -------------------------
model = FNO1d(modes=16, width=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# -------------------------
# Train
# -------------------------
epochs = 1000
trained_model = train_fno(model, dataloader, optimizer, scheduler, device, epochs, t_grid)
