import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.fft import fft, ifft
from datetime import datetime
from data_fno import MultiFunctionDatasetODE, custom_collate_ODE_fn

# -------------------------
# Spectral Convolution Layer (No Change)
# -------------------------

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bim, iom -> bom", input, weights)

    def forward(self, x):
        B, C, N = x.shape
        x_ft = fft(x, dim=-1)
        out_ft = torch.zeros(B, self.out_channels, N, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)
        x = ifft(out_ft, dim=-1).real
        return x


# -------------------------
# Fourier Layer with Dropout and Skip Connections
# -------------------------

class FourierLayerWithDropoutAndSkip(nn.Module):
    def __init__(self, width, modes, dropout_rate=0.1):
        super(FourierLayerWithDropoutAndSkip, self).__init__()
        self.fourier_conv = SpectralConv1d(width, width, modes)
        self.local_conv = nn.Conv1d(width, width, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(width)  # Normalize across the 'width' dimension

    def forward(self, x):
        residual = x
        x = self.fourier_conv(x)
        x = self.local_conv(x)
        x = self.dropout(x)
        
        # Apply LayerNorm over the width dimension (axis 1)
        x = self.layer_norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # Ensure proper dimension for LayerNorm

        return x + residual


# -------------------------
# FNO1D Without Positional Encoding
# -------------------------

class FNO1D(nn.Module):
    def __init__(self, modes=[16, 32, 64], width=64, input_channels=2, output_channels=1, hidden_size=128, dropout_rate=0.1):
        super(FNO1D, self).__init__()
        self.width = width
        self.fc0 = nn.Linear(input_channels, width)

        # Multi-scale Fourier layers with different numbers of modes
        self.layers = nn.ModuleList([FourierLayerWithDropoutAndSkip(width, mode, dropout_rate) for mode in modes])
        self.act = nn.Tanh()

        self.fc1 = nn.Linear(width, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_channels)

    def forward(self, u, t):
        # Directly use u and t without positional encoding
        input_tensor = torch.stack([u, t], dim=-1)  # [B, N, 2]
        x = self.fc0(input_tensor)  # [B, N, width]
        x = x.permute(0, 2, 1)  # [B, width, N]

        for layer in self.layers:
            residual = x
            x = layer(x)
            x = self.act(x)
            x = x + residual  # Skip connection

        x = x.permute(0, 2, 1)  # [B, N, width]
        x = self.fc2(self.fc1(x))  # Output projection
        return x.squeeze(-1)  # [B, N]


# -------------------------
# Compute Loss Function (No Change)
# -------------------------

def compute_loss(model, u, t):
    x = model(u, t)  # shape [B, N]
    dx = torch.autograd.grad(
        outputs=x.sum(),  # ensure scalar output for autograd
        inputs=t,
        create_graph=True
    )[0]  # shape [B, N]

    residual = dx + x - u
    physics_loss = torch.mean(residual ** 2)

    x0 = x[:, 0]  # approximate x(0) by first time step
    initial_loss = torch.mean((x0 - 1.0) ** 2)

    return physics_loss, initial_loss


# -------------------------
# Training Function (With ReduceLROnPlateau Scheduler)
# -------------------------

def train(model, dataloader, optimizer, scheduler, num_epochs=1000, plot=True):
    for epoch in range(num_epochs):
        physics_loss_total = 0
        initial_loss_total = 0

        model.train()
        for u, t, _, _ in dataloader:
            t = t.T.repeat(u.shape[0], 1)
            t.requires_grad_(True)

            physics_loss, initial_loss = compute_loss(model, u, t)
            physics_loss_total += physics_loss
            initial_loss_total += initial_loss

            loss = 30*physics_loss + initial_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute average loss for the epoch
        avg_physics_loss = physics_loss_total/len(dataloader)
        avg_initial_loss = initial_loss_total/len(dataloader)
        avg_loss = avg_physics_loss + avg_initial_loss

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss.item():.4f}, '
              f'Physics Loss: {avg_physics_loss.item():.6f}, Initial Loss: {avg_initial_loss.item():.6f}, '
              f'Time: {datetime.now().time()}')

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            model_filename = f'epochs_[{epoch+1}]_model_time_[{timestamp}]_loss_[{avg_loss.item():.4f}].pth'
            torch.save(model.state_dict(), f"trained_models/fno/{model_filename}")

        scheduler.step(avg_loss)  # Step scheduler based on validation loss

    return model


# -------------------------
# Hyperparameters and Model Initialization
# -------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_functions = 200000
grf_lb = 0.02
grf_ub = 0.5
end_time = 1.0
num_domain = 200
num_initial = 20
batch_size = 512
m = 200

# Instantiate and load dataset
dataset = MultiFunctionDatasetODE(
    m=m,
    n_functions=n_functions,
    function_types=['grf', 'linear', 'sine', 'polynomial', 'constant'],
    end_time=end_time,
    num_domain=num_domain,
    num_initial=num_initial,
    grf_lb=grf_lb,
    grf_ub=grf_ub
)

dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_ODE_fn, shuffle=True)

# Model setup
model = FNO1D(modes = [64, 64, 64], width=64, input_channels=2, 
              output_channels=1, hidden_size=128, dropout_rate=0.05)
model.to(device)

# Optimizer and scheduler
lr = 0.001
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Start training
trained_model = train(model, dataloader, optimizer, scheduler, num_epochs=1000)