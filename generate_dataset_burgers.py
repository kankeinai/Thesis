from utils.data_burgers import Burgers1D  # assuming it's in utils/data_burgers.py
import numpy as np
import random
import torch

# ---------------------------------------------------------------------
# Set random seeds for reproducibility
# ---------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------
# 1) Create the dataset in memory (expensive step happens only once)
# ---------------------------------------------------------------------
ds_test = Burgers1D(
    n_samples=100000,
    Nx=64,
    Nt=100,
    nu=0.05,    
    smoothing= False,
    save_path="datasets/burgers1d/",
    fraction_supervised=0,
    include_supervision=True,
    name="train",
    solver="ivp",
    control_functions=["grf", "sine", "fourier", "step"],
)
