from utils.data_burgers import save_burgers_dataset, load_burgers_dataset, BurgerEquationDatasetFNO, custom_collate_fno_fn
from torch.utils.data import DataLoader
import numpy as np
import random
import torch


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
ds_train = BurgerEquationDatasetFNO(
    n_samples=2000,
    Nx=64,
    Nt=200,
    function_types=['sine', 'grf', 'poly', 'bump', 'fourier', 'mixed'],
    nu_range=(0.01, 0.05),
    include_supervision=True,
    fraction_supervised=1,
)

# ---------------------------------------------------------------------
# 2) Save it                                                           
# ---------------------------------------------------------------------
save_burgers_dataset(ds_train, path="datasets/burgers/", name="test")

