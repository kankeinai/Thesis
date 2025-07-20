import torch
from utils.scripts import solve_optimization

# Usage example
from models.fno import FNO1d
from models.lno import LNO1d, LNO1d_extended
from models.deeponet import DeepONetCartesianProd
# from models.deeponet import DeepONet  # For DeepONet

m = 200
problem = 'nonlinear'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


modes = 16
width = 64
depth = 5
hidden_layer = 128

model = FNO1d(
    modes=modes,
    width=width,
    depth=depth,
    activation="silu",
    hidden_layer=hidden_layer
).to(device)

ckpt = torch.load("trained_models/nonlinear/fno/unsupervised/best/best_model.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
#linear deep o net [5, 1, 0, 0.001]
#nonlinear 

#linear fno

# Now solve for optimal u
solve_optimization(
    model, problem,
    lr=0.001,
    architecture="fno",  # or "deeponet"
    w=[2, 1, 1, 0, 0],
    num_epochs=30000,
    bounds = [-1.5, 1.5],
    m=m,
    device=device
)