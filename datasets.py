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

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



problem_1 = lambda dx, x, u, t: dx + x - u
problem_2 = lambda dx, x, u, t: dx - np.cos(4*np.pi*t) - u
problem_3 = lambda dx, x, u, t: dx - u
problem_4 = lambda dx, x, u, t: dx - 5/2*(-x +x*u -u**2)
problem_5 = lambda dx, x, u, t: dx - x**2 - u

problems = [problem_1, problem_2, problem_3, problem_4, problem_5]
problem_name = ['linear', 'oscillatory', 'polynomial_tracking', 'nonlinear', 'singular_arc']
idx = 0


architecture = 'fno'
print("Starting to create dataset")
ds = MultiFunctionDatasetODE(
    m=200, 
    n_functions=100000,
    function_types=['grf', 'polynomial', 'sine', 'linear', 'constant'],
    grf_lb= -3,
    grf_ub= 3,
    architecture=architecture,
    end_time=1,
    project=False,
    bound = [-1, 1],
    num_domain=200,
    include_supervision=True,
    fraction_supervised=0.2,
    problem=problem_name[idx],
)
print("Starting to save dataset")
path = save_dataset(ds, 'train', problem_name[idx], SEED)

