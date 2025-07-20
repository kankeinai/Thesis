import torch
from utils.scripts import solve_optimization
import random
import numpy as np
# Usage example
from models.lno import LNO1d_extended, LNO1d
from utils.settings import compute_loss_uniform_grid, datasets, gradient_finite_difference
from utils.scripts import load_data


def relative_error(prediction, target, dim=None, eps=1e-8):
    num = torch.sqrt(torch.sum((prediction - target) ** 2, dim=dim))
    den = torch.sqrt(torch.sum(target ** 2, dim=dim)).clamp(min=eps)
    return num / den

# Assume you've set up everything as in your script...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 69
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# from models.deeponet import DeepONet  # For DeepONet

m = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        
problems = ['linear', 'oscillatory', 'polynomial_tracking', 'nonlinear', 'singular_arc']
idx = 4
compute_loss = compute_loss_uniform_grid[problems[idx]]

architecture = 'lno'

train_loader, test_loader = load_data(
    architecture,
    'datasets/singular_arc/train-date-2025-07-19-sine.pt',
    'datasets/singular_arc/test-date-2025-07-19-sine.pt',
    SEED,
)
modes = 32
width = 4
hidden_layer = 128

# Instantiate and load the model as needed
model = LNO1d(modes=modes, width=width, activation="sine",  batch_norm=True, active_last=True, hidden_layer=hidden_layer).to(device)
ckpt = torch.load("trained_models/singular_arc/lno/attempt_started20250720_133633/epoch[320]_model_time_[20250720_133633]_loss_[0.0307].pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])

model.eval()

all_errors = []

with torch.no_grad():
    for batch in test_loader:
        # Assuming your test_loader yields: u, t, trajectory, mask (adjust if needed)
        if architecture == "deeponet":
            u, t, t0, ut, trajectory, mask = batch
        else:
            u, t, trajectory, mask = batch

        u = u.to(device)
        t = t.to(device)
        trajectory = trajectory.to(device)

        # Predict on uniform grid (for fair comparison)
        pred = model(u, t)
        # Compute relative error for each trajectory in the batch
        errors = relative_error(pred, trajectory, dim=1)  # dim=1: per-trajectory
        all_errors.append(errors.cpu())

all_errors = torch.cat(all_errors)
mean_err = all_errors.mean().item()
std_err = all_errors.std().item()

print(f"Mean relative test error: {mean_err:.4f} ± {std_err:.4f}")


