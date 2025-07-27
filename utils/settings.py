import torch
import numpy as np

compute_loss_uniform_grid = {
    'linear': {
        'physics_loss': lambda args: torch.mean((args['dx_dt'] + args['x'] - args['u'])**2),
        'initial_loss': lambda args: (args['x'] - 1.0).pow(2).mean(),
        'boundary_loss':lambda args: 0,
    },
    'oscillatory': {
        'physics_loss': lambda args: torch.mean((args['dx_dt']  - args['u'] - torch.cos(4*torch.pi*args['t']))**2),
        'initial_loss': lambda args: (args['x'] - 0).pow(2).mean(),
        'boundary_loss': lambda args: (args['x'] - 0).pow(2).mean(),
    },
    'polynomial_tracking': {
        'physics_loss': lambda args: torch.mean((args['dx_dt'] - args['u'])**2),
        'initial_loss': lambda args: (args['x'] - 0).pow(2).mean(),
        'boundary_loss':lambda args: 0,
    },
    'nonlinear': {
        'physics_loss': lambda args: torch.mean((args['dx_dt'] - 5/2*( - args['x'] + args['x']* args['u'] - args['u']**2))**2),
        'initial_loss': lambda args: (args['x'] - 1).pow(2).mean(),
        'boundary_loss':lambda args: 0,
    },
    'singular_arc': {
        'physics_loss': lambda args: torch.mean((args['dx_dt'] - (args['x']**2 +  args['u']))**2),
        'initial_loss': lambda args: (args['x'] - 1).pow(2).mean(),
        'boundary_loss': lambda args: (args['x'] - 0).pow(2).mean(),
    }
}

compute_loss_random_grid = {
    'linear': {
        'physics_loss': lambda args: torch.mean((args['dx_dt'] + args['x'] - args['u'])**2),
        'initial_loss': lambda args: (args['x'] - torch.ones_like(args['x'])).pow(2).mean()
    },
    'oscillatory': {
        'physics_loss': lambda args: torch.mean((args['dx_dt']  - args['u'] - torch.cos(4*torch.pi*args['t']))**2),
        'initial_loss': lambda args: (args['x'] - torch.zeros_like(args['x'])).pow(2).mean(),
        'boundary_loss': lambda args: (args['x'] - torch.zeros_like(args['x'])).pow(2).mean(),
    },
    'polynomial_tracking': {
        'physics_loss': lambda args: torch.mean((args['dx_dt'] - args['u'])**2),
        'initial_loss': lambda args: (args['x'] - torch.zeros_like(args['x'])).pow(2).mean(),
    },
    'nonlinear': {
        'physics_loss': lambda args: torch.mean((args['dx_dt'] - 5/2*( - args['x'] + args['x']* args['u'] - args['u']**2))**2),
        'initial_loss': lambda args: (args['x'] - torch.ones_like(args['x'])).pow(2).mean(),
    },
    'singular_arc': {
        'physics_loss': lambda args: torch.mean((args['dx_dt'] - (args['x']**2 +  args['u']))**2),
        'initial_loss': lambda args: (args['x'] - torch.ones_like(args['x'])).pow(2).mean(),
        'boundary_loss': lambda args: (args['x'] - torch.zeros_like(args['x'])).pow(2).mean(),
    }
}

datasets = {
    'linear': {
        'train' : 'datasets/linear/train.pt',
        'validation' : 'datasets/linear/validation.pt'
    },  
    'oscillatory': {
        'train': 'datasets/oscillatory/train.pt',
        'validation': 'datasets/oscillatory/validation.pt'
    },
    'polynomial_tracking': {
        'train': 'datasets/polynomial_tracking/train.pt',
        'validation': 'datasets/polynomial_tracking/validation.pt'
    },
    'nonlinear': {
        'train': 'datasets/nonlinear/train.pt',
        'validation': 'datasets/nonlinear/validation.pt'
    },
    'singular_arc': {
        'train': 'datasets/singular_arc/train.pt',
        'validation': 'datasets/singular_arc/validation.pt'
    }
}

optimal_solutions = {
    'linear':{
        'x' : lambda args: (np.sqrt(2) * np.cosh(np.sqrt(2) * (args['t'] - 1)) - np.sinh(np.sqrt(2) * (args['t'] - 1)))/(np.sqrt(2) * np.cosh(np.sqrt(2)) + np.sinh(np.sqrt(2))),
        'u' : lambda args: (np.sinh(np.sqrt(2) * (args['t'] - 1)))/( np.sqrt(2) * np.cosh(np.sqrt(2)) + np.sinh(np.sqrt(2))),
    },
   'oscillatory': {
    'x': lambda args: (
        (4 * np.pi / (16 * np.pi**2 + 1)) * np.sin(4 * np.pi * args['t'])
    ),
    'u': lambda args: (
        -1 / (16 * np.pi**2 + 1) * np.cos(4 * np.pi * args['t'])
    )
},
    'polynomial_tracking': {
    'x': lambda args: -0.8862 * np.exp(args['t']) - 1.1138 * np.exp(-args['t']) + args['t']**2 + 2,
    'u': lambda args: -0.8862 * np.exp(args['t']) + 1.1138 * np.exp(-args['t']) + 2 * args['t'],
},
    'nonlinear':{
        'x' : lambda args: 4/(1+3*np.exp(5*args['t']/2)),
        'u' : lambda args: 2/(1+3*np.exp(5*args['t']/2)),
    },
    'singular_arc':{
        'x' : lambda args: (1-args['t'])/(1+args['t']),
        'u' : lambda args: -2/(1+args['t'])**2 - ( (1-args['t'])/(1+args['t']))**2,
    }
}

objective_functions = {
    'linear':  lambda args: 1/2*torch.trapz((args['x']**2 + args['u']**2).squeeze(), args['t'].squeeze()) ,
    'oscillatory': lambda args: 1/2*torch.trapz((args['x']**2 + args['u']**2).squeeze(), args['t'].squeeze()) ,
    'polynomial_tracking': lambda args: torch.trapz(((args['x']-args['t']**2)**2 + args['u']**2).squeeze(), args['t'].squeeze()),
    'nonlinear': lambda args: -args['x'][-1],
    'singular_arc': lambda args: torch.mean(torch.trapz((args['u']**2).squeeze(), args['t'].squeeze())) 
}


def gradient_automatic(x, t):
    batch_size = x.shape[0]
    n_points = t.shape[0]
    dim_x = x.shape[1]
     # Physics loss
    dx = torch.zeros(batch_size, n_points, dim_x, device=t.device)
    # This loop is a bottleneck but i havent found a way to parallize this efficiently
    for b in range(batch_size):
        # Compute gradients for each batch independently
        dx[b] = torch.autograd.grad(x[b], t, torch.ones_like(x[b]), create_graph=True)[0]

    dx_dt = dx[:,:,0]

    return dx_dt

def gradient_finite_difference(x, t):
    dx_dt,  = torch.gradient(x, spacing=(t[0, :],), dim=1)
    return dx_dt.squeeze(-1)

trained_models ={
    'linear': {
        'deeponet': 'trained_models/linear/deeponet/unsupervised/epoch[1800]_model_time_[20250717_215634]_loss_[0.0006].pth',
        'fno': 'trained_models/linear/fno/unsupervised/epoch[180]_model_time_[20250718_201619]_loss_[0.0027].pth',
        'lno': 'trained_models/linear/lno/unsupervised/epoch[670]_model_time_[20250718_232320]_loss_[0.0011].pth',
    },
    'nonlinear': {
        'deeponet': 'trained_models/nonlinear/deeponet/unsupervised/epoch[1500]_model_time_[20250717_215941]_loss_[0.0099].pth',
        'fno': 'trained_models/nonlinear/fno/unsupervised/epoch[620]_model_time_[20250718_211507]_loss_[0.0010].pth',
        'lno': 'trained_models/nonlinear/lno/unsupervised/epoch[360]_model_time_[20250719_151537]_loss_[0.0597].pth',
    },
    'oscillatory': {
        'deeponet': 'trained_models/oscillatory/deeponet/unsupervised/epoch[1500]_model_time_[20250717_222232]_loss_[0.0065].pth',
        'fno': 'trained_models/oscillatory/fno/unsupervised/epoch[770]_model_time_[20250718_205034]_loss_[0.0028].pth',
        'lno': 'trained_models/oscillatory/lno/unsupervised/epoch[1000]_model_time_[20250718_232530]_loss_[0.0017].pth'
    },
    'polynomial_tracking': {
        'deeponet': 'trained_models/polynomial_tracking/deeponet/unsupervised/epoch[1300]_model_time_[20250717_222408]_loss_[0.0149].pth',
        'fno': 'trained_models/polynomial_tracking/fno/unsupervised/epoch[780]_model_time_[20250718_210320]_loss_[0.0050].pth',
        'lno': 'trained_models/polynomial_tracking/lno/unsupervised/epoch[1000]_model_time_[20250718_233449]_loss_[0.0127].pth',
    },
    'singular_arc': {
        'deeponet': 'trained_models/singular_arc/deeponet/unsupervised/epoch[1500]_model_time_[20250717_220059]_loss_[0.0059].pth',
        'fno': 'trained_models/singular_arc/fno/unsupervised/epoch[230]_model_time_[20250718_215927]_loss_[0.0091].pth',
        'lno': 'trained_models/singular_arc/lno/attempt_started20250720_215514/epoch[1000]_model_time_[20250720_215514]_loss_[0.0148].pth',
    }
}

architecture_settings = {
    'fno': {
        'linear': {
            'modes': 16,
            'width': 32,
            'depth': 4,
            'hidden_layer': 128,
            'activation': 'silu',
        },
        'nonlinear': {
            'modes': 16,
            'width': 64,
            'depth': 5,
            'hidden_layer': 128,
            'activation': 'silu',
        },
        'oscillatory': {
            'modes': 16,
            'width': 32,
            'depth': 4,
            'hidden_layer': 64,
            'activation': 'silu',
        },
        'polynomial_tracking': {
            'modes': 16,
            'width': 32,
            'depth': 4,
            'hidden_layer': 64,
            'activation': 'silu',
        },
        'singular_arc': {
            'modes': 32,
            'width': 64,
            'depth': 6,
            'hidden_layer': 256,
            'activation': 'silu',
        },
    },
    'deeponet': {
        'branch_net' : [200, 200, 200, 200, 200, 200],
        'branch_activations' : ['tanh', 'tanh', 'tanh', 'tanh','none'],
        'trunk_net' : [1, 200, 200, 200, 200, 200],
        'trunk_activations' : ['tanh', 'tanh', 'tanh', 'tanh','none']
    },
    'lno': {
        'linear': {
            'extended' : False,
            'modes' : 8,
            'width' : 4,
            'hidden_layer' : 64,
            'activation' : 'silu',
            'batch_norm' : False,
            'active_last' : False,
        },
        'nonlinear': {
            'extended' : True,
            'modes' : [8, 8],
            'width' : 4,
            'hidden_layer' : 64,
            'depth' : 2,
            'activation' : 'tanh',
            'active_last' : True,
        },
        'oscillatory': {
            'extended' : False,
            'modes' : 8,
            'width' : 4,
            'hidden_layer' : 64,
            'activation' : 'silu',
            'batch_norm' : False,
            'active_last' : False,
        },
        'polynomial_tracking': {
            'extended' : False,
            'modes' : 8,
            'width' : 4,
            'hidden_layer' : 64,
            'activation' : 'silu',
            'batch_norm' : False,
            'active_last' : False,
        },
        'singular_arc': {
            'extended' : False,
            'modes' : 32,
            'width' : 16,
            'hidden_layer' : 128,
            'activation' : 'silu',
            'batch_norm' : True,
            'active_last' : True,
        }
    }
}

boundaries = {
    'linear': None,
    'oscillatory': None,
    'polynomial_tracking': None,
    'nonlinear': [-1.5, 1.5],
    'singular_arc': [-3.5, 0],
}



def compute_residual_burgers(u, f, dx, dt, nu=0.01):
    """
    Compute PDE residual: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x² + f(x)
    Soft BCs handled separately.
    """
    dudt = torch.gradient(u, spacing=dt, dim=2)[0]
    dudx = torch.gradient(u, spacing=dx, dim=1)[0]
    d2udx2 = torch.gradient(dudx, spacing=dx, dim=1)[0]

    f_expanded = f.unsqueeze(-1).expand_as(u)
    residual = dudt + u * dudx - nu * d2udx2 - f_expanded

    return residual
def dirichlet_boundary_loss(u):
    """
    Penalize boundary values at x=0 and x=1.
    u: [B, Nx, Nt]
    """
    left  = u[:, 0, :]     # y(0, t)
    right = u[:, -1, :]    # y(1, t)
    return torch.mean(left**2) + torch.mean(right**2)


def compute_residual_diffusion_reaction(u, f, dx, dt, nu=0.01, alpha=0.01):
    """
    Compute residual for the PDE:
        ∂u/∂t = ν ∂²u/∂x² - α u² + f(x)

    Inputs:
        u : [B, Nx, Nt]  – predicted solution
        f : [B, Nx]      – time-invariant forcing term (control)
    """
    dudt = torch.gradient(u, spacing=dt, dim=2)[0]  # [B, Nx, Nt]
    du_dx = torch.gradient(u, spacing=dx, dim=1)[0]
    d2udx2 = torch.gradient(du_dx, spacing=dx, dim=1)[0]  # [B, Nx, Nt]
    reaction = -alpha * u**2  # [B, Nx, Nt]
    f_expanded = f.unsqueeze(-1).expand_as(u)  # [B, Nx, Nt]
    residual = dudt - nu * d2udx2 + reaction - f_expanded  # [B, Nx, Nt]

    return residual

def compute_residual_heat(u, f, dx, dt, nu):
    """
    Compute residual for:
        ∂u/∂t = ν ∂²u/∂x² + f(x)

    Inputs:
        u: [B, Nx, Nt] – predicted solution
        f: [B, Nx]     – forcing term
    """
    dudt = torch.gradient(u, spacing=dt, dim=2)[0]  # [B, Nx, Nt]
    du_dx = torch.gradient(u, spacing=dx, dim=1)[0]
    d2u_dx2 = torch.gradient(du_dx, spacing=dx, dim=1)[0]  # [B, Nx, Nt]
    f_expanded = f.unsqueeze(-1).expand_as(u)  # [B, Nx, Nt]
    residual = dudt - nu * d2u_dx2 - f_expanded  # [B, Nx, Nt]
    return residual
