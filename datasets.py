import matplotlib.pyplot as plt
import numpy as np
from utils.data import MultiFunctionDatasetODE, DiskBackedODEDataset, save_dataset
from torch.utils.data import DataLoader
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
import random
import torch
from torch.utils.data import Dataset
import os
from datetime import datetime

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


problem_name = ['linear', 'oscillatory', 'polynomial_tracking', 'nonlinear', 'singular_arc']
idx = 1


architecture = 'fno'
print("Starting to create dataset")

ds = MultiFunctionDatasetODE(
    m=200, 
    n_functions=10000,
    function_types=['grf', 'polynomial'],
    grf_lb= 0.02,
    grf_ub= 0.5,
    architecture=architecture,
    degree_range = (1, 8),
    slope_range = (-2,2),
    intercept_range=(-2,2),
    end_time=1,
    project=False,
    bound = [-1, 1],
    num_domain=200,
    include_supervision=True,
    fraction_supervised=1,
    problem=problem_name[idx],
)
print("Starting to save dataset")
path = save_dataset(ds, 'test', problem_name[idx], SEED)

