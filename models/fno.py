import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    """
    One-dimensional spectral convolution using Fourier transforms.
    """
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, x_ft, weights):
        # x_ft: [B, in_channels, modes], weights: [in_channels, out_channels, modes]
        return torch.einsum("bim, iom -> bom", x_ft, weights)

    def forward(self, x):
        # x: [B, C, N]
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # [B, C, N//2+1]
        out_ft = torch.zeros(
            B, self.out_channels, x_ft.size(-1),
            dtype=x_ft.dtype, device=x_ft.device
        )
        # keep low-frequency modes
        modes = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], self.weights)
        x = torch.fft.irfft(out_ft, n=N, dim=-1)
        return x

class SimpleBlock1d(nn.Module):
    """
    Flexible FNO block: configurable depth, activation, dropout.
    """
    def __init__(self, modes, width, depth=4, activation="gelu", hidden_layer=128):
        super().__init__()
        self.width = width
        # select activation
        act = activation.lower()
        self.act = nn.SiLU() if act == "silu" else nn.GELU()

        # lift (u, t) -> feature space
        self.fc0 = nn.Linear(2, width)
        # spectral + 1x1 conv layers
        self.specs = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(depth)
        ])
        self.ws = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(depth)
        ])
        # projection MLP
        self.fc1 = nn.Linear(width, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 1)

    def forward(self, x):
        # x: [B, N, 2] concatenated (u, t)
        x = self.fc0(x)             # [B, N, width]
        x = x.permute(0, 2, 1)      # [B, width, N]
        for spec, w in zip(self.specs, self.ws):
            y = spec(x) + w(x)
            x = self.act(y)
        x = x.permute(0, 2, 1)      # [B, N, width]
        x = self.act(self.fc1(x))   # [B, N, 128]
        x = self.fc2(x).squeeze(-1)  # [B, N]
        return x                # global skip on u

class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator with refactored block.
    """
    def __init__(self, modes, width, depth=4, activation="gelu", hidden_layer=128):
        super().__init__()
        self.conv1 = SimpleBlock1d(
            modes=modes,
            width=width,
            depth=depth,
            activation=activation,
            hidden_layer=hidden_layer
        )

    def forward(self, u, t):
        # unchanged forward: concat forcing and time
        x = torch.cat([u.unsqueeze(-1), t.unsqueeze(-1)], dim=-1)
        return self.conv1(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
