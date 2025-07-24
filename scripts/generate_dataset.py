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
idx = 3

architecture = 'lno'
print("Starting to create dataset")

ds = MultiFunctionDatasetODE(
    m=200, 
    n_functions=10000,
    function_types=['sine', 'polynomial'],
    grf_lb= 0.05,
    grf_ub= 0.5,
    architecture=architecture,
    degree_range = (1, 5),
    slope_range = (-2, 2),
    intercept_range=(-2, 2),
    frequency_range=(0.1, 30),    # For 'sine'
    amplitude_range=(0.5, 2),  
    coeff_range=(-3, 3),
    end_time=1,
    project=True,
    bound = [-1.5, -1.5],
    num_domain=200,
    include_supervision=True,
    fraction_supervised=1,
    problem=problem_name[idx],
)

path = f"datasets/{problem_name[idx]}/"
print("Starting to save dataset")
path = save_dataset(ds,  path, name = 'test')

