import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    """
    One-dimensional spectral convolution layer using Fourier transforms.

    This layer transforms the input to the frequency domain, applies
    learnable complex weights on a fixed number of low-frequency modes,
    and transforms back to real space.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    modes1 : int
        Number of lowest Fourier modes to retain (higher modes are zeroed).
    """
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
        x_ft = torch.fft.rfft(x, dim=-1) 
        
        # after (out-of-place)
        head = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        # make a zero‐tensor for the trailing modes
        tail = torch.zeros(
            x_ft.size(0), x_ft.size(1), x_ft.size(2) - self.modes1,
            dtype=x_ft.dtype, device=x_ft.device
        )
        out_ft = torch.cat([head, tail], dim=-1)

        x = torch.fft.irfft(out_ft, n=N, dim=-1)  
        return x

class SimpleBlock1d(nn.Module):
    """
    A single Fourier Neural Operator (FNO) block in 1D, combining spectral convolutions
    with local 1×1 convolutions and pointwise MLP projections.

    Parameters
    ----------
    modes : int
        Number of low-frequency Fourier modes to retain in each spectral convolution.
    width : int
        Hidden channel dimension for feature processing.
    """
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

class FNO1d(nn.Module):
    """
    Wrapper module for a 1D Fourier Neural Operator (FNO).

    This class encapsulates a single SimpleBlock1d, exposing a streamlined
    interface for mapping an input field and auxiliary features to an output field.

    Parameters
    ----------
    modes : int
        Number of low-frequency Fourier modes to retain in each spectral convolution.
    width : int
        Hidden feature dimension inside the spectral blocks.
    """
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()
        self.conv1 = SimpleBlock1d(modes, width)

    def forward(self, u, t):
        x = torch.cat([u.unsqueeze(-1), t], dim=-1)
        return self.conv1(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)