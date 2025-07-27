from models.fno import FNO1d
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FNO1d(
        modes=32,
        width=64,
        depth=4,
        out_dim=100,    # Nt output points
        activation='gelu'
    ).to(device)

model.load_state_dict(torch.load("trained_models/heat1d/ckpt_ep0180.pt"))
model.eval()


import numpy as np
import matplotlib.pyplot as plt
baseline = np.load("baseline_solutions/heat_control_time_invariant.npz")

w = [10, 1, 1, 0, 0]
initial_guess = torch.rand((1, 64), dtype=torch.float32, device=device, requires_grad=True)

def y_desired(x, A=1.0, x0=0.5, sigma=0.1):
    return torch.tensor(
        A * np.exp(-((x - x0)**2) / (2 * sigma**2)),
        dtype=torch.float32,
        device=device
    )
x = np.linspace(0, 1, 64)
target_profile = y_desired(x)

def calculate_error(pred_y, pred_u, optimal_y, optimal_u):
    rel_err_u = torch.norm(pred_u - optimal_u) / torch.norm(optimal_u)

    rel_err_x = torch.norm(pred_y - optimal_y) / torch.norm(optimal_y)
    return rel_err_u.item(), rel_err_x.item()

def compute_residual_heat(y, u, dx, dt, nu):
    """
    y: [B, Nx, Nt]
    u: [B, Nx]
    """
    dudt = (y[:, :, 1:] - y[:, :, :-1]) / dt  # [B, Nx, Nt-1]
    d2ydx2 = (y[:, 2:, :-1] - 2 * y[:, 1:-1, :-1] + y[:, :-2, :-1]) / dx**2  # [B, Nx-2, Nt-1]
    u_expanded = u[:, 1:-1].unsqueeze(-1).expand_as(d2ydx2)  # match shape
    
    residual = dudt[:, 1:-1, :] - nu * d2ydx2 - u_expanded  # physics residual
    return residual


import torch
import numpy as np
import torch.optim as optim

num_epochs = 30000

if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Setup
t_np = np.linspace(0, 1, 100)
x_np = np.linspace(0, 1, 64)
dx = x_np[1] - x_np[0]
dt = t_np[1] - t_np[0]
x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0) 

# Trainable control u(x)
u_param = torch.nn.Parameter(initial_guess)  # shape: [1, Nx]
optimizer = optim.Adam([u_param], lr=1e-3)

# Baseline reference
optimal_y = torch.tensor(baseline['y'], dtype=torch.float32, device=device)  # [Nx, Nt]
optimal_u = torch.tensor(baseline['u'], dtype=torch.float32, device=device)   # [Nx]

analytics = {
    'obj': [],
    'rel_err_u': [],
    'rel_err_y': [],
    'total_loss': []
}
def compute_residual_heat(y, u, dx, dt, nu=0.01):
    """
    Compute residual of 1D heat equation: ∂y/∂t = ν ∂²y/∂x² + u(x)

    y: [B, Nx, Nt]
    u: [B, Nx]
    """
    # ∂y/∂t (along time dim 2)
    dudt = torch.gradient(y, spacing=dt, dim=2)[0]  # [B, Nx, Nt]

    # ∂²y/∂x² (second spatial derivative along dim 1)
    dy_dx = torch.gradient(y, spacing=dx, dim=1)[0]
    d2y_dx2 = torch.gradient(dy_dx, spacing=dx, dim=1)[0]

    # Expand u to match [B, Nx, Nt]
    u_expanded = u.unsqueeze(-1).expand_as(y)

    # Residual
    return dudt - nu * d2y_dx2 - u_expanded
def initial_and_boundary_loss(y):
    ic_loss = torch.mean(y[:, :, 0]**2)
    bc_loss = torch.mean(y[:, 0, :]**2) + torch.mean(y[:, -1, :]**2)
    return ic_loss + bc_loss

def objective_heat_control(y, u, target, rho=1e-3):
    final_y = y[:, :, -1]
    tracking_loss = torch.mean((final_y - target.unsqueeze(0))**2)
    control_loss = torch.mean(u**2)
    return tracking_loss/2 + rho * control_loss/2

# Optimization loop
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()
    
    u_clamped = u_param  # [1, Nx]
    y = model(u_clamped, x)  # [1, Nx, Nt]
    

    obj = objective_heat_control(y, u_clamped, target_profile)
    residual = compute_residual_heat(y, u_clamped, dx, dt)
    initial_loss = torch.mean(y[:, :, 0]**2)
    boundary_loss = torch.mean(y[:, 0, :]**2) + torch.mean(y[:, -1, :]**2)

    loss = obj + 5*torch.mean(residual**2) + initial_loss + boundary_loss

    analytics['obj'].append(obj.item())
    analytics['total_loss'].append(loss.item())

    loss.backward()
    optimizer.step()

    rel_err_u, rel_err_y = calculate_error(
        y.detach(), u_param.detach(), optimal_y.view(1, 64, 100), optimal_u
    )

    analytics['rel_err_u'].append(rel_err_u)
    analytics['rel_err_y'].append(rel_err_y)

    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | rel_err_u: {rel_err_u:.4f}, rel_err_y: {rel_err_y:.4f}")


from models.fno import FNO1d
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FNO1d(
         modes=32,
        width=64,
        in_dim=2,  # u(x), x
        out_dim=100,  # Nt
        depth=5,
        activation="gelu",
    ).to(device)

model.load_state_dict(torch.load("trained_models/diffusion1d/ckpt_ep0300.pt"))
model.eval()


def compute_residual_diffusion_reaction(y, u, dx, dt, nu=0.01, alpha=0.01):
    """
    Compute PDE residual for: ∂y/∂t = ν ∂²y/∂x² - α y² + u(x)

    y: [B, Nx, Nt]
    u: [B, Nx] — time-invariant control
    """
    dudt = torch.gradient(y, spacing=dt, dim=2)[0]            # ∂y/∂t
    dy_dx = torch.gradient(y, spacing=dx, dim=1)[0]           # ∂y/∂x
    d2y_dx2 = torch.gradient(dy_dx, spacing=dx, dim=1)[0]     # ∂²y/∂x²
    y_sq = y**2                                               # nonlinear reaction term
    u_expanded = u.unsqueeze(-1).expand_as(y)

    return dudt - nu * d2y_dx2 + alpha * y_sq - u_expanded


def y_desired(x, A=1.0, sigma=0.05):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    bump1 = A * torch.exp(-((x - 0.3)**2) / (2 * sigma**2))
    bump2 = A * torch.exp(-((x - 0.7)**2) / (2 * sigma**2))
    return bump1 + bump2


x = np.linspace(0, 1, 64)
target_profile = y_desired(x)



import torch
import numpy as np
import torch.optim as optim

num_epochs = 30000

if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Setup
t_np = np.linspace(0, 1, 100)
x_np = np.linspace(0, 1, 64)
dx = x_np[1] - x_np[0]
dt = t_np[1] - t_np[0]
x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0) 
initial_guess = torch.rand((1, 64), dtype=torch.float32, device=device, requires_grad=True)


# Trainable control u(x)
u_param = torch.nn.Parameter(initial_guess)  # shape: [1, Nx]
optimizer = optim.Adam([u_param], lr=1e-3)

analytics = {
    'obj': [],
    'total_loss': []
}

def initial_and_boundary_loss(y):
    ic_loss = torch.mean(y[:, :, 0]**2)
    bc_loss = torch.mean(y[:, 0, :]**2) + torch.mean(y[:, -1, :]**2)
    return ic_loss + bc_loss

def objective_heat_control(y, u, target, rho=1e-3):
    final_y = y[:, :, -1]
    tracking_loss = torch.mean((final_y - target.unsqueeze(0))**2)
    control_loss = torch.mean(u**2)
    return tracking_loss/2 + rho * control_loss/2

# Optimization loop
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()
    
    u_clamped = u_param  # [1, Nx]
    y = model(u_clamped, x)  # [1, Nx, Nt]
    

    obj = objective_heat_control(y, u_clamped, target_profile)
    residual = compute_residual_diffusion_reaction(y, u_clamped, dx, dt)
    initial_loss = torch.mean(y[:, :, 0]**2)
    boundary_loss = torch.mean(y[:, 0, :]**2) + torch.mean(y[:, -1, :]**2)

    loss = obj + 5*torch.mean(residual**2) + initial_loss + boundary_loss

    analytics['obj'].append(obj.item())
    analytics['total_loss'].append(loss.item())

    loss.backward()
    optimizer.step()


    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}| {torch.mean(residual**2).item():.6f}")


plt.plot(u_clamped.detach().cpu().numpy()[0], label='Optimized Control u(x)', color='blue')


from models.fno import FNO1d
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FNO1d(
         modes=32,
        width=64,
        in_dim=2,  # u(x), x
        out_dim=100,  # Nt
        depth=5,
        activation="gelu",
    ).to(device)

model.load_state_dict(torch.load("trained_models/burgers1d/best.pt"))
model.eval()


import numpy as np
def shock_target(x, location=0.5, steepness=50):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    return (1.0 / (1.0 + torch.exp(-steepness * (x - location)))) * 0.1

x_np = np.linspace(0, 1, 64)
target_profile = shock_target(x_np)  # shape: [64]


def compute_residual_burgers(u, f, dx, dt, nu=0.01):
    dudt = torch.gradient(u, spacing=dt, dim=2)[0]
    dudx = torch.gradient(u, spacing=dx, dim=1)[0]
    d2udx2 = torch.gradient(dudx, spacing=dx, dim=1)[0]
    f_expanded = f.unsqueeze(-1).expand_as(u)
    return dudt + u * dudx - nu * d2udx2 - f_expanded

def soft_boundary_loss(u):
    return torch.mean(u[:, 0, :]**2) + torch.mean(u[:, -1, :]**2)

def initial_condition_loss(u):
    return torch.mean(u[:, :, 0]**2)

def objective_shock_tracking(u, target_profile):
    final_u = u[:, :, -1]
    return 0.5 * torch.mean((final_u - target_profile.unsqueeze(0))**2)



# Hyperparameters
Nx, Nt = 64, 100
x_np = np.linspace(0, 1, Nx)
t_np = np.linspace(0, 1, Nt)
dx = x_np[1] - x_np[0]
dt = t_np[1] - t_np[0]

x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
t_tensor = torch.tensor(t_np, dtype=torch.float32, device=device).unsqueeze(0)

initial_guess = torch.rand(1, Nx, device=device) * 1.5
u_param = torch.nn.Parameter(initial_guess)
optimizer = torch.optim.Adam([u_param], lr=1e-3)

target_profile = shock_target(x_np)

for epoch in range(1, 5001):
    optimizer.zero_grad()
    u_pred = model(u_param, x_tensor)  # shape: [1, Nx, Nt]

    res = compute_residual_burgers(u_pred, u_param, dx, dt, nu=0.01)

    loss = (
        20 * objective_shock_tracking(u_pred, target_profile)
      + 5  * torch.mean(res**2)
      + 1.0  * initial_condition_loss(u_pred)
      + 1.0  * soft_boundary_loss(u_pred)
      + 0.01 * torch.mean(u_param**2)
    )

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}| phys {torch.mean(res**2)}")

