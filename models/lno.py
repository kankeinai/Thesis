import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class PR(nn.Module):
    # ... (Your PR definition remains unchanged)
    def __init__(self, in_channels, out_channels, modes1):
        super(PR, self).__init__()
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        self.weights_pole = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def output_PR(self, lambda1, alpha, weights_pole, weights_residue):   
        Hw = torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],lambda1.shape[0], device=alpha.device, dtype=torch.cfloat)
        term1 = torch.div(1, torch.sub(lambda1, weights_pole))
        Hw = weights_residue * term1
        Pk = -Hw
        output_residue1 = torch.einsum("bix,xiok->box", alpha, Hw) 
        output_residue2 = torch.einsum("bix,xiok->bok", alpha, Pk) 
        return output_residue1, output_residue2    

    def forward(self, x, t):
        dt = (t[1] - t[0]).item()   
        alpha = torch.fft.fft(x)
        lambda0 = torch.fft.fftfreq(t.shape[0], dt)*2*np.pi*1j
        lambda1 = lambda0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1 = lambda1.to(x.device, dtype=torch.cfloat)
        output_residue1, output_residue2 = self.output_PR(lambda1, alpha, self.weights_pole, self.weights_residue)
        x1 = torch.fft.ifft(output_residue1, n=x.size(-1))
        x1 = torch.real(x1)
        x2 = torch.zeros(output_residue2.shape[0], output_residue2.shape[1], t.shape[0], device=alpha.device, dtype=torch.cfloat)    
        term1 = torch.einsum("bix,kz->bixz", self.weights_pole, t.type(torch.complex64).reshape(1,-1))
        term2 = torch.exp(term1) 
        x2 = torch.einsum("bix,ioxz->boz", output_residue2, term2)
        x2 = torch.real(x2)
        x2 = x2 / x.size(-1)
        return x1 + x2

    
class LNO1d_extended(nn.Module):
    """
    LNO1d with arbitrary depth, stacking (PR + Conv1d + BatchNorm + SiLU) blocks.
    """
    def __init__(self, width, modes, activation="sine", hidden_layer=128, depth=1, active_last=True):
        super().__init__()
        self.width = width
        self.modes = modes
        self.depth = depth

        self.fc0 = nn.Linear(2, self.width)

        # Stack PR+Conv1d+BatchNorm blocks
        self.operator_blocks = nn.ModuleList([
            nn.ModuleDict({
                'pr': PR(self.width, self.width, self.modes[i]),
                'conv': nn.Conv1d(self.width, self.width, 1),
                'norm': nn.BatchNorm1d(self.width)
            }) for i in range(depth)
        ])

        if activation == "sine":
            self.act = torch.sin
        elif activation == "tanh":
            self.act = torch.tanh
        elif activation == "silu":
            self.act = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.fc1 = nn.Linear(self.width, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 16)
        self.fc3 = nn.Linear(16, 1)
        self.active_last = active_last

    def forward(self, x, t_grid):
        # Input lifting: concatenate x and t, project
        x = torch.cat((x.unsqueeze(-1), t_grid.unsqueeze(-1)), dim=-1)  # (batch, N, 2)
        x = self.fc0(x)  # (batch, N, width)
        x = x.permute(0, 2, 1)  # (batch, width, N)

        # t_grid must be shape (N,)
        t = t_grid.squeeze(-1)[0, :]

        # Stacked operator blocks with nonlinearity after each
        for block in self.operator_blocks:
            x1 = block['pr'](x, t)
            x2 = block['conv'](x)
            x = x1 + x2
            x = block['norm'](x)
            if self.active_last:
                x = self.act(x)

        # Final MLP head
        x = x.permute(0, 2, 1)  # (batch, N, width)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)  # (batch, N, 1)
        x = self.act(x)
        x = self.fc3(x)  # (batch, N, 1)

        return x.squeeze(-1)

class LNO1d(nn.Module):
    """
    Old-notation LNO1d: depth=1, explicit block names.
    """
    def __init__(self, width, modes, activation="silu", hidden_layer=128, batch_norm = False, active_last=False):
        super().__init__()

        self.width = width
        self.modes1 = modes
        self.active_last = active_last

        self.fc0 = nn.Linear(2, self.width)            # w0.weight, w0.bias
        self.conv0 = PR(self.width, self.width, self.modes1)    # pr0.weights_pole, pr0.weights_residue
        self.w0 = nn.Conv1d(self.width, self.width, 1)     # conv0.weight, conv0.bias
        if batch_norm:
            self.bn = nn.BatchNorm1d(self.width)
        self.batch_norm = batch_norm

        # Activation
        if activation == "sine":
            self.act = torch.sin
        elif activation == "tanh":
            self.act = torch.tanh
        elif activation == "silu":
            self.act = F.silu
        else:
            raise ValueError("Unsupported activation: " + activation)

        self.fc1 = nn.Linear(self.width, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 1)

    def forward(self, x, t_grid):
        # x: (batch, N), t_grid: (batch, N)
        x = torch.cat((x.unsqueeze(-1), t_grid.unsqueeze(-1)), dim=-1)  # (batch, N, 2)
        x = self.fc0(x)           # (batch, N, width)
        x = x.permute(0, 2, 1)    # (batch, width, N)
        t = t_grid.squeeze(-1)[0, :]  # (N,)

        # Operator block 0
        x1 = self.conv0(x, t)
        x2 = self.w0(x)

        x = x1 + x2

        if self.batch_norm:
            x = self.bn(x)

        if self.active_last:
            x = self.act(x)

        # Final MLP head
        x = x.permute(0, 2, 1)  # (batch, N, width)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.squeeze(-1)