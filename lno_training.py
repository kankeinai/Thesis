import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import time
from timeit import default_timer
import time
from data_fno import MultiFunctionDatasetODE, custom_collate_ODE_fn 
from torch.utils.data import DataLoader
from datetime import datetime


class PR(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(PR, self).__init__()

        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        self.weights_pole = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
   
    def output_PR(self, lambda1,alpha, weights_pole, weights_residue):   
        Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],lambda1.shape[0], device=alpha.device, dtype=torch.cfloat)
        term1=torch.div(1,torch.sub(lambda1,weights_pole))
        Hw=weights_residue*term1
        Pk=-Hw  # for ode, Pk equals to negative Hw
        output_residue1=torch.einsum("bix,xiok->box", alpha, Hw) 
        output_residue2=torch.einsum("bix,xiok->bok", alpha, Pk) 
        return output_residue1,output_residue2    
    

    def forward(self, x):

        t=grid_x_train.cuda()
        #Compute input poles and resudes by FFT
        dt=(t[1]-t[0]).item()
        alpha = torch.fft.fft(x)
        lambda0=torch.fft.fftfreq(t.shape[0], dt)*2*np.pi*1j
        lambda1=lambda0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1=lambda1.cuda()
        start=time.time()

        # Obtain output poles and residues for transient part and steady-state part
        output_residue1,output_residue2= self.output_PR(lambda1, alpha, self.weights_pole, self.weights_residue)

        # Obtain time histories of transient response and steady-state response
        x1 = torch.fft.ifft(output_residue1, n=x.size(-1))
        x1 = torch.real(x1)
        x2=torch.zeros(output_residue2.shape[0],output_residue2.shape[1],t.shape[0], device=alpha.device, dtype=torch.cfloat)    
        term1=torch.einsum("bix,kz->bixz", self.weights_pole, t.type(torch.complex64).reshape(1,-1))
        term2=torch.exp(term1) 
        x2=torch.einsum("bix,ioxz->boz", output_residue2,term2)
        x2=torch.real(x2)
        x2=x2/x.size(-1)
        return x1+x2

class LNO1d(nn.Module):
    def __init__(self, width, modes, hidden_layer=256):
        super(LNO1d, self).__init__()

        self.width = width
        self.modes1 = modes
        self.fc0 = nn.Linear(2, self.width) 

        self.conv0 = PR(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 1)

    def forward(self,x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = torch.sin(x)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x =  torch.sin(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
    
def compute_loss(model, u, t):
    
    x = model(u)

    dt = (t[1]-t[0]).item()

    dx_dt = (x[:, 1:, :] - x[:, :-1, :]) / dt  # crude finite diff
    residual = dx_dt + x[:, :-1, :] - u[:, :-1, :]

    physics_loss = torch.mean(residual ** 2)

    x0 = x[:, 0, :]
    initial_loss = torch.mean((x0 - 1.0)**2)

    return physics_loss, initial_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Dataset parameters
n_functions = 1000000
grf_lb = 0.02
grf_ub = 0.5
end_time = 1.0
num_domain = 200
num_initial = 20
batch_size = 1024
m = 200

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

dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_ODE_fn, shuffle=True)

learning_rate = 0.001
epochs = 100
step_size = 100
gamma = 0.9

modes = 32
width = 4

# model
model = LNO1d(width,modes).cuda()

# ====================================
# Training 
# ====================================
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
start_time = time.time()

timestop = 0

for ep in range(timestop, epochs):
    model.train()
    t1 = default_timer()
    epoch_loss = 0.0
    n_batches = 0

    for u, t, _, _ in dataloader:
        u = u.to(device).unsqueeze(-1)
        grid_x_train = t
        optimizer.zero_grad()
        physics_loss, initial_loss = compute_loss(model, u, grid_x_train)
        loss = physics_loss + initial_loss
        loss.backward()
        optimizer.step()
        print(f"Epoch: {ep}, Physics loss: {physics_loss}, Initial loss: {initial_loss}")
        epoch_loss += loss.item()
        n_batches += 1

    epoch_loss /= n_batches
    # Save model checkpoint
    if (ep + 1) % 10 == 0:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        model_filename = f'epochs_[{ep+1}]_model_time_[{timestamp}]_loss_[{epoch_loss:.4f}].pth'
        torch.save(model.state_dict(), f"trained_models/lno/{model_filename}")

    scheduler.step()
