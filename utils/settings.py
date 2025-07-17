import torch

compute_loss_uniform_grid = {
    'linear': {
        'physics_loss': lambda args: torch.mean((args['dx_dt'] + args['x'] - args['u'])**2),
        'initial_loss': lambda args: (args['x'-1] - 1.0).pow(2).mean()
    },
    'oscillatory': {
        'physics_loss': lambda args: torch.mean((args['dx_dt']  - args['u'] - torch.cos(4*torch.pi*args['t']))**2),
        'initial_loss': lambda args: (args['x'] - 0).pow(2).mean(),
    },
    'polynomial_tracking': {
        'physics_loss': lambda args: torch.mean((args['dx_dt'] - args['u'])**2),
        'initial_loss': lambda args: (args['x'] - 0).pow(2).mean(),
    },
    'nonlinear': {
        'physics_loss': lambda args: torch.mean((args['dx_dt'] - 5/2*( - args['x'] + args['x']* args['u'] - args['u']**2))**2),
        'initial_loss': lambda args: (args['x'] - 1).pow(2).mean(),
    },
    'singular_arc': {
        'physics_loss': lambda args: torch.mean((args['dx_dt'] - (args['x']**2 +  args['u']))**2),
        'initial_loss': lambda args: (args['x'] - 1).pow(2).mean(),
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
    }
}