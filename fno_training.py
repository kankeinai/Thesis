import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.fft import fft, ifft
from functools import reduce, partial
from datetime import datetime
import operator
from data_fno import MultiFunctionDatasetODE, custom_collate_ODE_fn
import os


# -------------------------
# Spectral Convolution Layer
# -------------------------
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # shape [B, C, N//2 + 1]
        out_ft = torch.zeros(B, self.out_channels, x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=N, dim=-1)  # Return to physical space
        return x


# -------------------------
# Simple Block with GELU
# -------------------------
class SimpleBlock1d(nn.Module):
    def __init__(self, modes, width):
        super(SimpleBlock1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)

        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------
# Wrapper Model
# -------------------------
class Net1d(nn.Module):
    def __init__(self, modes, width):
        super(Net1d, self).__init__()
        self.conv1 = SimpleBlock1d(modes, width)

    def forward(self, u, t):
        x = torch.cat([u.unsqueeze(-1), t], dim=-1)
        return self.conv1(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# -------------------------
# Physics-Informed Loss Function
# -------------------------
def compute_loss(model, u, t, dt=1/200, lambda_phys=1.0, lambda_init=10.0):
    x = model(u, t)
    dx_dt = (x[:, 1:, :] - x[:, :-1, :]) / dt
    residual = dx_dt + x[:, :-1, :] - u[:, :-1].unsqueeze(-1)
    physics_loss = torch.mean(residual ** 2)
    initial_loss = torch.mean((x[:, 0, :] - 1.0) ** 2)
    return lambda_phys * physics_loss + lambda_init * initial_loss

# -------------------------
# Training Function
# -------------------------
def train(model, dataloader, optimizer, scheduler, t_grid, num_epochs=1000):
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for u, _, _, _ in dataloader:
            u = u.to(device)
            t = t_grid[:u.shape[0], :].to(device)
            loss = compute_loss(model, u, t)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.6f}, Time: {datetime.now().time()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Time: {datetime.now().time()}')

        
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        os.makedirs("trained_models/fno", exist_ok=True)
        torch.save(model.state_dict(), f"trained_models/fno/model_epoch{epoch+1}_{timestamp}_loss{avg_loss:.4f}.pth")

        scheduler.step(avg_loss)

    return model

# -------------------------
# Hyperparameters and Setup
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_functions = 1000000
grfs = ['grf', 'linear', 'sine', 'polynomial', 'constant']
end_time = 1.0
m = 200
batch_size = 1024
num_epochs = 1000

# Time grid
t_grid = torch.linspace(0, end_time, m).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)

# Dataset and Dataloader
dataset = MultiFunctionDatasetODE(
    m=m,
    n_functions=n_functions,
    function_types=grfs,
    end_time=end_time,
    num_domain=m,
    num_initial=20,
    grf_lb=0.02,
    grf_ub=0.5
)

dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_ODE_fn, shuffle=True)

# Model and optimizer
model = Net1d(modes=16, width=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Train
trained_model = train(model, dataloader, optimizer, scheduler, t_grid=t_grid, num_epochs=num_epochs)
