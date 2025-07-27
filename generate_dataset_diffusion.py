from utils.data_diffusion import Diffusion1D # assuming it's in heat_dataset.py
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
ds_train = Diffusion1D(
    n_samples=2000,
    Nx=64,
    Nt=100,
    nu=0.01,   # or 'sine', 'fourier', 'step'
    smoothing=False,
    project=False,          # preserve amplitude variability
    save_path="datasets/diffusion1d/",
    fraction_supervised=1,  # 50% supervised
    include_supervision=True,
    name = "test",

)
