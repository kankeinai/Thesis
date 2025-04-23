import torch
from utils.data import MultiFunctionDatasetODE, custom_collate_ODE_fn_fno 
from torch.utils.data import DataLoader
from datetime import datetime
from models.lno import *
from utils.scripts import *
import os

# -------------------------
# Problem specific loss function
# -------------------------
def compute_loss(model, u, t):
    
    x = model(u, t)
    dt = (t[1]-t[0]).item()
    dx_dt = (x[:, 1:, :] - x[:, :-1, :]) / dt
    residual = dx_dt + x[:, :-1, :] - u[:, :-1, :]
    physics_loss = torch.mean(residual ** 2)
    x0 = x[:, 0, :]
    initial_loss = torch.mean((x0 - 1.0)**2)

    return physics_loss, initial_loss

def train_lno(model, dataloader, device, epochs, folder = "trained_models/lno", learning_rate=0.001, step_size = 100, gamma = 0.9):

    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for u, t, _, _ in dataloader:
            u = u.to(device).unsqueeze(-1)
            optimizer.zero_grad()
            physics_loss, initial_loss = compute_loss(model, u, t)
            loss = physics_loss + initial_loss
            loss.backward()
            optimizer.step()
            print(f"Epoch: {ep}, Physics loss: {physics_loss}, Initial loss: {initial_loss}")
            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= n_batches

        if (ep + 1) % 10 == 0:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            os.makedirs(folder, exist_ok=True)
            model_filename = f'epochs_[{ep+1}]_model_time_[{timestamp}]_loss_[{epoch_loss:.4f}].pth'
            torch.save(model.state_dict(), folder+f"/{model_filename}")

        scheduler.step()

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

# ====================================
# Model definition
# ====================================
modes = 32
width = 4
model = LNO1d(width,modes).cuda()

# ====================================
# Training 
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
trained_model = train_lno(model, dataloader, optimizer, scheduler, device, epochs, learning_rate, step_size, gamma)