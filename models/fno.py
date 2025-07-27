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
    def __init__(self, modes, width, depth=4, activation="gelu", hidden_layer=128, in_dim=2, out_dim=1):

        super().__init__()
        self.width = width
        # select activation
        act = activation.lower()
        self.act = nn.SiLU() if act == "silu" else nn.GELU()

        # lift (u, t) -> feature space
        self.fc0 = nn.Linear(in_dim, width)
        # spectral + 1x1 conv layers
        self.specs = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(depth)
        ])
        self.ws = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(depth)
        ])
        # projection MLP
        self.fc1 = nn.Linear(width, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, out_dim)

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
    def __init__(self, modes, width, depth=4, in_dim = 2, out_dim=1, activation="gelu", hidden_layer=128):
        super().__init__()
        self.conv1 = SimpleBlock1d(
            modes=modes,
            width=width,
            depth=depth,
            activation=activation,
            hidden_layer=hidden_layer,
            in_dim=in_dim,
            out_dim=out_dim
        )

    def forward(self, u, t, y1 = None):
        # unchanged forward: concat forcing and time
        u = u.unsqueeze(-1)  # [B, Nx, 1]
        t = t.unsqueeze(-1)  # [B, Nx, 1]

        if y1 is not None:
            x = torch.cat([u, t, y1], dim=-1)  # [B, Nx, 2 + k]
        else:
            x = torch.cat([u, t], dim=-1)      # [B, Nx, 2]

        return self.conv1(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
def add_padding2(x, num_pad1, num_pad2):
    if max(num_pad1) > 0 or max(num_pad2) > 0:
        res = F.pad(x, (num_pad2[0], num_pad2[1], num_pad1[0], num_pad1[1]), 'constant', 0.)
    else:
        res = x
    return res

def remove_padding2(x, num_pad1, num_pad2):
    if max(num_pad1) > 0 or max(num_pad2) > 0:
        res = x[..., num_pad1[0]:-num_pad1[1], num_pad2[0]:-num_pad2[1]]
    else:
        res = x
    return res


def _get_act(act):
    if act == 'tanh':
        func = F.tanh
    elif act == 'gelu':
        func = F.gelu
    elif act == 'relu':
        func = F.relu_
    elif act == 'elu':
        func = F.elu_
    elif act == 'leaky_relu':
        func = F.leaky_relu_
    else:
        raise ValueError(f'{act} is not supported')
    return func

@torch.jit.script
def compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    res =  torch.einsum("bixy,ioxy->boxy", a, b)
    return res

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2, 3])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, device=x.device,
                                dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), dim=[2, 3])
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,
                 width=64, fc_dim=128,
                 layers=None,
                 in_dim=3, out_dim=1,
                 act='gelu', 
                 pad_ratio=[0., 0.]):
        super(FNO2d, self).__init__()
        """
        Args:
            - modes1: list of int, number of modes in first dimension in each layer
            - modes2: list of int, number of modes in second dimension in each layer
            - width: int, optional, if layers is None, it will be initialized as [width] * [len(modes1) + 1] 
            - in_dim: number of input channels
            - out_dim: number of output channels
            - act: activation function, {tanh, gelu, relu, leaky_relu}, default: gelu
            - pad_ratio: list of float, or float; portion of domain to be extended. If float, paddings are added to the right. 
            If list, paddings are added to both sides. pad_ratio[0] pads left, pad_ratio[1] pads right. 
        """
        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, 'Cannot add padding in more than 2 directions'
        self.modes1 = modes1
        self.modes2 = modes2
    
        self.pad_ratio = pad_ratio
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * (len(modes1) + 1)
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2d(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, layers[-1])
        self.fc3 = nn.Linear(layers[-1], out_dim)
        self.act = _get_act(act)

    def forward(self, x):
        '''
        Args:
            - x : (batch size, x_grid, y_grid, 2)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        '''
        size_1, size_2 = x.shape[1], x.shape[2]
        if max(self.pad_ratio) > 0:
            num_pad1 = [round(i * size_1) for i in self.pad_ratio]
            num_pad2 = [round(i * size_2) for i in self.pad_ratio]
        else:
            num_pad1 = num_pad2 = [0.]

        length = len(self.ws)
        batchsize = x.shape[0]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)   # B, C, X, Y
        x = add_padding2(x, num_pad1, num_pad2)
        size_x, size_y = x.shape[-2], x.shape[-1]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y)
            x = x1 + x2
            if i != length - 1:
                x = self.act(x)
        x = remove_padding2(x, num_pad1, num_pad2)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x
