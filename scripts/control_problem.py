import torch
from utils.scripts import solve_optimization

# Usage example
from models.fno import FNO1d
from models.lno import LNO1d, LNO1d_extended
from models.deeponet import DeepONetCartesianProd
# from models.deeponet import DeepONet  # For DeepONet

m = 200
problem = 'singular_arc'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model Parameters
m = 200         # sensor size (branch input size)
n_hid = 200     # layer's hidden sizes
p = 200         # output size
dim_x = 1       # trunk (trunk input size)

# Specify the MLP architecture
branch_net = [m, n_hid,  n_hid, n_hid, n_hid, p]
branch_activations = ['tanh', 'tanh', 'tanh', 'tanh','none']
trunk_net = [dim_x, n_hid,  n_hid, n_hid, n_hid, p]
trunk_activations = ['tanh', 'tanh', 'tanh', 'tanh','none']
model = DeepONetCartesianProd(branch_net, trunk_net, branch_activations, trunk_activations)
model.to(device)

ckpt = torch.load("trained_models/singular_arc/deeponet/unsupervised/epoch[1500]_model_time_[20250717_220059]_loss_[0.0059].pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
#linear deep o net [5, 1, 0, 0.001]
#nonlinear 

#linear fno

# Now solve for optimal u
solve_optimization(
    model, problem,
    lr=0.001,
    architecture="deeponet",  # or "deeponet"
    w=[100, 1, 1, 0, 10],
    num_epochs=30000,
    bounds = [-3.5, 0],
    m=m,
    device=device
)