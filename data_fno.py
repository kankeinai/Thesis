import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import griddata

def generate_gaussian_random_field_1d(grid_size, length_scale, end_time=1.0, mean=0, variance=1):

    # Create a grid of spatial frequencies, scaled to the [0, end_time] domain
    kx = np.fft.fftfreq(grid_size, d=end_time / grid_size)  # Adjust frequency scaling
    k_squared = kx**2  # Compute the squared frequency values

    # Avoid division by zero for k = 0
    k_squared[0] = np.inf

    # Construct the power spectral density (PSD) using the scaled length scale
    psd = variance * np.exp(-k_squared * (length_scale**2))

    # Generate a random field in Fourier space
    random_field = np.fft.ifft(np.sqrt(psd) * (np.random.normal(size=grid_size) 
                                               + 1j * np.random.normal(size=grid_size)))

    # Transform back to spatial domain and adjust mean
    field = np.real(random_field)
    field = mean + (field - np.mean(field)) * (np.sqrt(variance) / np.std(field))
    
    return field

def project_to_range(data, new_min=-1, new_max=1):
    # Min-Max normalization
    min_val = np.min(data)
    max_val = np.max(data)
    min_max = np.maximum(np.abs(min_val),np.abs(max_val))

    if min_max <=1.0:
        return data
    else:
        proj_coef = np.random.uniform(0,(1/min_max),1)
        return proj_coef*data

class MultiFunctionDatasetODE(Dataset):

    def __init__(self, m, n_functions, function_types=['grf', 'linear', 'sine'],end_time=1,num_domain=900,num_initial=100,grf_ub=None,grf_lb=None,project=False):
        self.m = m
        self.n_functions = n_functions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.time_domain = np.linspace(0, end_time, m)
        self.function_types = function_types
        self.end_time = end_time
        self.project = project
        
        # domain points
        self.num_domain = num_domain
        self.num_initial = num_initial

        # generate different grfs with different length scales
        self.grf_lb = grf_lb
        self.grf_ub = grf_ub


        # Pre-generate all the data
        self.data = []
        for _ in range(n_functions):
            func_type = np.random.choice(self.function_types)  # Randomly select a function type
            input_function = self.generate_function(func_type)
            self.data.append(input_function)
    

    def generate_function(self, func_type):
        if func_type == 'grf':
            grid_size = np.random.randint(self.m // 2, self.m * 2)

            if self.grf_lb is not None and self.grf_ub is not None:
                length_scale = np.random.uniform(self.grf_lb, self.grf_ub, 1)
            else:
                length_scale = self.length_scale

            grf = generate_gaussian_random_field_1d(grid_size, length_scale, end_time=self.end_time)
            grid_points = np.linspace(0, self.end_time, grid_size)
            values = np.interp(self.time_domain, grid_points, grf)

        elif func_type == 'linear':
            slope = np.random.uniform(-2, 2)  # Random slope
            intercept = np.random.uniform(-1, 1)  # Random intercept
            values = slope * self.time_domain + intercept

        elif func_type == 'sine':
            frequency = np.random.uniform(0.1, 10)  # Random frequency
            amplitude = np.random.uniform(0.5, 2)  # Random amplitude
            phase = np.random.uniform(0, 2 * np.pi)  # Random phase
            values = amplitude * np.sin(2 * np.pi * frequency * self.time_domain + phase)

        elif func_type == 'polynomial':
            coefficients = np.random.uniform(-3, 3, size=np.random.randint(3, 8))  # Random polynomial coefficients
            values = np.polyval(coefficients, self.time_domain)

        elif func_type == 'constant':
            constant_value = np.random.uniform(-3, 3)  # Random constant value
            values = np.full_like(self.time_domain, constant_value)

        else:
            raise ValueError(f"Unsupported function type: {func_type}")

        return values


    def __len__(self):
        return self.n_functions

    def __getitem__(self, idx):
        input_function= self.data[idx]
        
        if self.project:
            input_function = self.project_to_range(input_function)
        

        return (
            input_function,
            self.end_time,
            self.num_domain,
            self.num_initial,
            self.time_domain,
            self.m,
        )

def custom_collate_ODE_fn(batch):

    time_domain = batch[0][4]   # fixed linspace grid
    m = batch[0][5]
    batch_size = len(batch)

    input_functions = np.zeros((batch_size, m))  
 
    for b, item in enumerate(batch):
        input_function = item[0]
        input_functions[b, :] = input_function

    return (
        torch.tensor(input_functions).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),  # [B, m]
        torch.from_numpy(np.copy(time_domain)).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),  # [1, m]
        None,
        None,
    )