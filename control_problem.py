import torch
from utils.scripts import solve_optimization

# Usage example
from models.fno import FNO1d
# from models.deeponet import DeepONet  # For DeepONet

m = 200
problem = 'linear'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate and load the model as needed
model = FNO1d(modes=32, width=64, depth=4, activation="silu", hidden_layer=128)
ckpt = torch.load("trained_models/linear/fno/your_model_path_here.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])

# Now solve for optimal u
solve_optimization(
    model, problem,
    lr=0.001,
    architecture="fno",  # or "deeponet"
    w=[100, 1],
    bounds=None,
    num_epochs=1000,
    m=m,
    device=device
)