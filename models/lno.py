import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class PR(nn.Module):
    """
    Pole-Residue layer for frequency-domain response modeling.

    Learns complex poles and residues to decompose the system response
    into transient and steady-state parts via rational approximation.
    """
    def __init__(self, in_channels, out_channels, modes1):
        """
        Initialize the PR block.

        Parameters
        ----------
        in_channels : int
            Number of input feature channels.
        out_channels : int
            Number of output feature channels.
        modes1 : int
            Number of frequency modes (poles/residues) to learn.
        """
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
    
    def forward(self, x, t):

        dt = (t[1] - t[0]).item()   
        alpha = torch.fft.fft(x)
        lambda0=torch.fft.fftfreq(t.shape[0], dt)*2*np.pi*1j
        lambda1=lambda0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1=lambda1.to(x.device, dtype=torch.cfloat)

        # Obtain output poles and residues for transient part and steady-state part
        output_residue1,output_residue2 = self.output_PR(lambda1, alpha, self.weights_pole, self.weights_residue)

        # Obtain time histories of transient response and steady-state response
        x1 = torch.fft.ifft(output_residue1, n=x.size(-1))
        x1 = torch.real(x1)
        x2=torch.zeros(output_residue2.shape[0],output_residue2.shape[1], t.shape[0], device=alpha.device, dtype=torch.cfloat)    
        term1=torch.einsum("bix,kz->bixz", self.weights_pole, t.type(torch.complex64).reshape(1,-1))
        term2=torch.exp(term1) 
        x2=torch.einsum("bix,ioxz->boz", output_residue2, term2)
        x2=torch.real(x2)
        x2=x2/x.size(-1)

        return x1+x2

class LNO1d(nn.Module):
    """
    One-dimensional Lagrangian Neural Operator (LNO) combining
    local MLP lift, frequency-domain PR block, and pointwise conv.
    """
    def __init__(self, width, modes, hidden_layer=256):
        """
        Initialize the LNO1d model.

        Parameters
        ----------
        width : int
            Hidden feature dimension after input lifting.
        modes : int
            Number of frequency modes for the PR block.
        hidden_layer : int, optional
            Size of the hidden layer in the final MLP.
        """
        super(LNO1d, self).__init__()

        self.width = width
        self.modes1 = modes
        self.fc0 = nn.Linear(2, self.width) 
        self.conv0 = PR(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x, t_grid):
        """
        Forward pass of the LNO1d.

        Parameters
        ----------
        x : torch.Tensor
            Input field, shape (batch, N).
        t : torch.Tensor
            Time grid, shape (N,) or (batch, N).

        Returns
        -------
        torch.Tensor
            Predicted field, shape (batch, N, 1).
        """
        x = torch.cat((x.unsqueeze(-1), t_grid.unsqueeze(-1)), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        t = t_grid.squeeze(-1)[0, :]

        x1 = self.conv0(x, t)
        x2 = self.w0(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)

        return x.squeeze(-1)
    