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
        'x' : lambda args: 0,
        'u' : lambda args: 0,
    },
    'oscillatory':{
        'x' : lambda args: 0,
        'u' : lambda args: 0,
    },
    'polynomial_tracking':{
        'x' : lambda args: 0,
        'u' :lambda args: 0,
    },
    'nonlinear':{
        'x' : lambda args: 0,
        'u' : lambda args: 0,
    },
    'singular_arc':{
        'x' : lambda args: 0,
        'u' : lambda args: 0,
    }
}

objective_functions = {
    'linear':  lambda args: 0,
    'oscillatory': lambda args: 0,
    'polynomial_tracking': lambda args: 0,
    'nonlinear': lambda args: 0,
    'singular_arc': lambda args: 0
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

