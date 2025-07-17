import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import os
from datetime import datetime

def custom_collate_ODE_fn(batch, architecture='deeponet'):
    """
    Custom collate function for ODE datasets that handles different architectures
    and both supervised/unsupervised samples.
    
    Args:
        batch: List of samples from the dataset
        architecture: Neural operator architecture ('deeponet', 'fno', 'lno')
        include_supervision: Whether supervision is enabled
    
    Returns:
        Collated batch suitable for the specified architecture, including boolean mask
    """

    
    # Extract common parameters
    end_time = batch[0][1]
    num_domain = batch[0][2]
    num_initial = batch[0][3]
    time_domain = batch[0][4]
    m = batch[0][5]

    batch_size = len(batch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_functions = np.array([item[0] for item in batch])

    if architecture == 'fno' or architecture == 'lno':
        time_points = np.linspace(0, end_time, m)
        # Repeat time_points for each sample in the batch
        time_points_batch = np.tile(time_points, (batch_size, 1))  # [batch_size, num_domain]
        result = (
            torch.tensor(input_functions).float().to(device),
            torch.tensor(time_points_batch).float().to(device),
        )
    elif architecture == 'deeponet':
        time_points = np.random.uniform(0, end_time, (num_domain, 1))
        initial_points = np.zeros((num_initial, 1))
        final_points = np.full((num_initial, 1), end_time)
        input_at_times = np.zeros((batch_size, num_domain))
        
        for b, item in enumerate(batch):
            input_function = item[0]
            input_at_time = np.interp(time_points.flatten(), time_domain, input_function)
            input_at_times[b, :] = input_at_time
            
        result = (
            torch.tensor(input_functions).float().to(device),
            torch.tensor(time_points).float().to(device),
            torch.tensor(initial_points).float().to(device),
            torch.tensor(input_at_times).float().to(device),

        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Use 'fno', 'lno', or 'deeponet'")
    
        # Build the mask as before
    mask = torch.tensor([item[-1] for item in batch],
                        dtype=torch.bool,
                        device=device)

    # Build the trajectories tensor, handling both numpy arrays and Tensors:
    traj_list = []
    for i, item in enumerate(batch):
        tr = item[-2]
        if mask[i]:
            if isinstance(tr, np.ndarray):
                t_t = torch.from_numpy(tr).float()
            elif isinstance(tr, torch.Tensor):
                t_t = tr.float()
            else:
                raise TypeError(f"Unexpected trajectory type: {type(tr)}")
        else:
            t_t = torch.zeros(m, dtype=torch.float32)
        traj_list.append(t_t.to(device))

    trajectories = torch.stack(traj_list, dim=0)  # [batch_size, m]
        # 3) Append to your result in one go:
    result += (trajectories, mask)                        

    return result

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


class ODEProblem:
    """
    Class to manage ODE problem configurations for optimal control problems.
    """
    
    def __init__(self, dynamics_func, initial_condition, solver_method='RK45', 
                 description=None):
        """
        Initialize an ODE problem.
        
        Args:
            dynamics_func: Function defining the ODE dynamics f(x, u, t)
            initial_condition: Initial condition x(0)
            solver_method: ODE solver method ('RK45', 'BDF', etc.)
            description: Human-readable description of the problem
        """
        self.dynamics_func = dynamics_func
        self.initial_condition = initial_condition
        self.solver_method = solver_method
        self.description = description or f"Custom ODE: {dynamics_func.__name__}"
    
    def solve_trajectory(self, control_func, t_span=(0, 1), t_eval=None):
        """
        Solve the ODE for a given control function.
        
        Args:
            control_func: Control function u(t)
            t_span: Time span (t0, tf)
            t_eval: Time points for evaluation
            
        Returns:
            Solution trajectory
        """
        def ode_system(t, x):
            u = control_func(t)
            return self.dynamics_func(x, u, t)
        
        # Use BDF solver for stiff problems, RK45 for non-stiff
        if self.solver_method == 'BDF':
            solver_kwargs = {'method': 'BDF', 'max_step': 0.01}
        elif self.solver_method == 'DOP853':
            solver_kwargs = {'method': 'DOP853', 'atol': 1e-10, 'rtol': 1e-8, 'max_step': 0.005}
        elif self.solver_method == 'LSODA':
            solver_kwargs = {
                'method': "LSODA",  'atol': 1e-6, 'rtol': 1e-9
            }
            
        else:
            solver_kwargs = {'method': self.solver_method}
        
        
        solution = solve_ivp(
            ode_system, 
            t_span, 
            [self.initial_condition], 
            t_eval=t_eval,
            **solver_kwargs
        )

        # Diagnostic metrics
      
        if not solution.success:
            raise RuntimeError(f"ODE solver failed: {solution.message}")
            
        return solution.y.flatten()
            
    
    def __str__(self):
        return f"ODEProblem: {self.description}"
    
    def __repr__(self):
        return self.__str__()


class ODEProblemRegistry:
    """
    Registry for predefined ODE problems.
    """
    
    def __init__(self):
        self._problems = {}
        self._register_default_problems()
    
    def _register_default_problems(self):
        """Register the default set of ODE problems."""
        
        linear_problem = ODEProblem(
            dynamics_func=lambda x, u, t: -x + u,
            initial_condition=1.0,
            solver_method='DOP853',
            description="Linear ODE: dx/dt = -x + u, x(0) = 1",
        )
        self.register('linear', linear_problem)
        
     
        oscillatory_problem = ODEProblem(
            dynamics_func=lambda x, u, t: np.cos(4*np.pi*t) + u,
            initial_condition=0.0,
            solver_method='DOP853',
            description="Oscillatory ODE: dx/dt = cos(4*pi*t) + u, x(0) = 0"
        )
        self.register('oscillatory', oscillatory_problem)
        

        polynomial_problem = ODEProblem(
            dynamics_func = lambda x, u, t:  u,
            initial_condition=0,
            solver_method='DOP853',
            description="Polynomial tracking: dx/dt = u"
        )
        self.register('polynomial_tracking', polynomial_problem)
        

        nonlinear_problem = ODEProblem(
            dynamics_func = lambda x, u, t: 5/2*(-x +x*u -u**2),
            initial_condition=1.0,
            solver_method='DOP853',
            description="Nonlinear ODE: dx/dt = 5/2*(-x +x*u -u**2), x(0) = 1"
        )
        self.register('nonlinear', nonlinear_problem)
        
        singular_problem = ODEProblem(
            dynamics_func = lambda x, u, t: x**2 + u,
            initial_condition=1.0,
            solver_method='DOP853',
            description="Singular arc: dx/dt = x**2 + u, x(0) = 1",
        )
        self.register('singular_arc', singular_problem)
    
    def register(self, name, problem):
        """Register a new ODE problem."""
        if not isinstance(problem, ODEProblem):
            raise ValueError("problem must be an ODEProblem instance")
        self._problems[name] = problem
    
    def get(self, name):
        """Get an ODE problem by name."""
        if name not in self._problems:
            raise ValueError(f"Unknown problem '{name}'. Available: {list(self._problems.keys())}")
        return self._problems[name]
    
    def list(self):
        """List all available problems."""
        return list(self._problems.keys())
    
    def create_custom(self, dynamics_func, initial_condition, solver_method='RK45', 
                     description=None):
        """Create a custom ODE problem."""
        return ODEProblem(dynamics_func, initial_condition, solver_method, description)


# Global registry instance
ode_registry = ODEProblemRegistry()


class MultiFunctionDatasetODE(Dataset):
    """
    Dataset class for generating control functions and optionally their corresponding 
    state trajectories for ODE-based optimal control problems.
    
    Supports both unsupervised (residual-only) and semi-supervised training paradigms
    as described in the PINO-control framework.
    """
    
    def __init__(self, m, n_functions, function_types=['linear', 'sine'], 
                 end_time=1, num_domain=900, num_initial=100, architecture='deeponet',
                 grf_ub=None, grf_lb=None, project=False, bound=None,
                 include_supervision=False, problem=None, fraction_supervised=None,
                 degree_range=(3, 8), coeff_range=(-3, 3), intercept_range=(-3, 3),
                 frequency_range=(0.1, 10), amplitude_range=(0.5, 2), phase_range=(0, 2 * np.pi),
                 slope_range=(-2, 2)):
        """
        Initialize the dataset.
        
        Args:
            m: Number of time discretization points
            n_functions: Number of control functions to generate
            function_types: List of function types to sample from
            end_time: End time of the time domain
            num_domain: Number of collocation points for physics residual
            num_initial: Number of points for initial condition enforcement
            grf_ub/grf_lb: Upper/lower bounds for GRF length scale
            project: Whether to project functions to bounded range
            include_supervision: Whether to compute ground-truth trajectories
            problem: ODEProblem instance or problem name string
            fraction_supervised: Fraction of samples to include supervision
        """
        self.m = m
        self.n_functions = n_functions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.time_domain = np.linspace(0, end_time, m)
        self.function_types = function_types
        self.end_time = end_time
        self.project = project
        self.degree_range = degree_range
        self.coeff_range = coeff_range
        self.slope_range = slope_range
        self.intercept_range = intercept_range
        self.architecture = architecture
        self.frequency_range = frequency_range
        self.amplitude_range = amplitude_range
        self.phase_range = phase_range

        # Store architecture for collate function selection
        self.architecture = architecture
        
        # Domain points for physics residual and initial conditions
        self.num_domain = num_domain
        self.num_initial = num_initial


        # Handle problem configuration
        if problem is not None:
            if isinstance(problem, str):
                # String name - get from registry
                if problem not in ode_registry.list():
                    available = ode_registry.list()
                    raise ValueError(f"Unknown problem '{problem}'. Available problems: {available}")
                self.problem = ode_registry.get(problem)
            elif isinstance(problem, ODEProblem):
                # Direct ODEProblem instance
                self.problem = problem
            else:
                raise ValueError("problem must be a string name or ODEProblem instance")
        else:
            raise ValueError("problem must be provided")

        if project:
            if not (isinstance(bound, (list, tuple)) and len(bound) == 2):
                raise ValueError("bound must be a list or tuple of length 2, e.g., [u_min, u_max]")
            if not all(isinstance(b, (int, float)) for b in bound):
                raise ValueError("bound values must be numeric (int or float)")
            self.bound = tuple(bound)

        # GRF parameters with validation
        if 'grf' in function_types:
            if grf_lb is None or grf_ub is None:
                raise ValueError("grf_lb and grf_ub must be provided when 'grf' is in function_types")
            if grf_lb >= grf_ub:
                raise ValueError("grf_lb must be less than grf_ub")
            
        self.grf_lb = grf_lb
        self.grf_ub = grf_ub

        # Semi-supervised parameters
        self.include_supervision = include_supervision

         # Pre-generate all the data
        self.data = []
        self.trajectories = [] 
        self.has_trajectory = np.zeros(n_functions, dtype=bool)
        
        # Validate fraction_supervised parameter
        if include_supervision:
            if fraction_supervised is None:
                raise ValueError("fraction_supervised must be provided when include_supervision=True")
            if not (0 < fraction_supervised <= 1):
                raise ValueError("fraction_supervised must be between 0 and 1")
            self.fraction_supervised = fraction_supervised
        else:
            self.fraction_supervised = 0.0  # No supervision
    
        for i in range(self.n_functions):
            func_type = np.random.choice(self.function_types)
            # Randomly decide if this function should have trajectory
            if self.include_supervision:
                should_supervise = np.random.random() < self.fraction_supervised
            else:
                should_supervise = False

            input_function, trajectory = self.generate_function(func_type, should_supervise)
            self.data.append(input_function)
            self.has_trajectory[i] = should_supervise
            self.trajectories.append(trajectory)
    

    def generate_function(self, func_type, should_supervise=False):
        """
        Generate a control function and optionally solve for its trajectory.
        This is more efficient than generating the function first and then solving.
        
        Args:
            func_type: Type of function to generate
            should_supervise: Whether to compute trajectory for this function
        """
        print(f"generating function {func_type}")
        # Generate the control function and its analytical form
        if func_type == 'grf':
            grid_size = np.random.randint(self.m // 2, self.m * 2)

            if self.grf_lb is not None and self.grf_ub is not None:
                length_scale = np.random.uniform(self.grf_lb, self.grf_ub, 1)
            else:
                length_scale = self.length_scale

            grf = generate_gaussian_random_field_1d(grid_size, length_scale, end_time=self.end_time)

            if self.project:
                grf = self._project_function(grf)

            grid_points = np.linspace(0, self.end_time, grid_size)
            # Interpolate GRF to control function with smooth derivatives
            control_func = CubicSpline(grid_points, grf)

            # Evaluate it on the standard time grid
            values = control_func(self.time_domain)


        elif func_type == 'linear':
            min_slope, max_slope = self.slope_range
            min_intercept, max_intercept = self.intercept_range
            slope = np.random.uniform(min_slope, max_slope)
            intercept = np.random.uniform(min_intercept, max_intercept)
            values = slope * self.time_domain + intercept

            if self.project:
                values = self._project_function(values)
                control_func =  lambda t: np.interp(t, self.time_domain, values)
            else:
                control_func = lambda t: slope * t + intercept

        elif func_type == 'sine':
            min_freq, max_freq = self.frequency_range
            min_amp, max_amp = self.amplitude_range
            min_phase, max_phase = self.phase_range

            frequency = np.random.uniform(min_freq, max_freq)
            amplitude = np.random.uniform(min_amp, max_amp)
            phase = np.random.uniform(min_phase, max_phase)
            values = amplitude * np.sin(2 * np.pi * frequency * self.time_domain + phase)

            if self.project:
                values = self._project_function(values)
                control_func = lambda t: np.interp(t, self.time_domain, values)
            else:
                control_func = lambda t: amplitude * np.sin(2 * np.pi * frequency * t + phase)

        elif func_type == 'polynomial':
            min_deg, max_deg = self.degree_range
            min_coeff, max_coeff = self.coeff_range
            degree = np.random.randint(min_deg, max_deg)
            coefficients = np.random.uniform(min_coeff, max_coeff, size=degree+1)
            values = np.polyval(coefficients, self.time_domain)

            if self.project:
                values = self._project_function(values)
                control_func = lambda t: np.interp(t, self.time_domain, values)
            else:
                control_func = lambda t: np.polyval(coefficients, t)

        elif func_type == 'constant':
            min_int, max_int = self.intercept_range
            constant_value = np.random.uniform(min_int, max_int)
            values = np.full_like(self.time_domain, constant_value)
            if self.project:
                values = self._project_function(values)
                control_func = lambda t: np.interp(t, self.time_domain, values)
            else:
                control_func = lambda t: constant_value

        else:
            raise ValueError(f"Unsupported function type: {func_type}")


        if should_supervise:
            trajectory = self.problem.solve_trajectory(control_func, t_eval=self.time_domain)
            return values, trajectory
        else:
            return values, None

    def solve_ode(self, control_func):
        """
        Solve the ODE for a given control function.
        This method is kept for backward compatibility but delegates to the problem.
        """
        return self.problem.solve_trajectory(control_func, t_eval=self.time_domain)

    def __len__(self):
        return self.n_functions

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            If include_supervision=False: (input_function, end_time, num_domain, num_initial, time_domain, m)
            If include_supervision=True: (input_function, end_time, num_domain, num_initial, time_domain, m, trajectory, has_trajectory) 
                                       or (input_function, end_time, num_domain, num_initial, time_domain, m, has_trajectory)
        """
        input_function = self.data[idx]

        # Base sample structure
        sample = (
            input_function, 
            self.end_time, 
            self.num_domain, 
            self.num_initial, 
            self.time_domain, 
            self.m, 
            self.trajectories[idx] if self.include_supervision else None, 
            self.has_trajectory[idx] if self.include_supervision else None
        )
        
        return sample

    def _project_function(self, function_values):
        """
        Project function values to the specified bounds.
        Adds variation for constant functions by mapping to a random value
        in the target range [u_min, u_max].

        Returns:
            projected: np.ndarray, projected function values
        """
        u_min, u_max = self.bound
        current_min = np.min(function_values)
        current_max = np.max(function_values)

        # Avoid division by zero or flat function issues
        if np.isclose(current_max, current_min):
            # Inject small random variation inside the bounds
            noise = np.random.uniform(u_min, u_max, size=function_values.shape)
            return noise

        # Otherwise, linearly rescale to fit in [u_min, u_max]
        scaled = (function_values - current_min) / (current_max - current_min)
        projected = u_min + scaled * (u_max - u_min)
        return projected


    def get_collate_fn(self):
        """
        Get the appropriate collate function based on the architecture.
        """

        return lambda batch: custom_collate_ODE_fn(batch, self.architecture)



class DiskBackedODEDataset(Dataset):
    def __init__(self, path, architecture='deeponet'):
        state = torch.load(path)
        self.values      = state['values']          # tensor [N, m]
        self.trajectories = state['trajectories']   # tensor [N, m]
        self.mask        = state['mask']            # bool tensor [N]
        self.time_domain = state['time_domain'].numpy()
        self.num_domain  = state['num_domain']
        self.num_initial = state['num_initial']
        self.end_time    = state['end_time']
        self.m           = state['m']
        self.architecture = architecture

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, idx):
        vals = self.values[idx]                      # tensor [m]
        traj = self.trajectories[idx] if self.mask[idx] else None
        mask = bool(self.mask[idx].item())
        return (
            vals, 
            self.end_time, 
            self.num_domain, 
            self.num_initial, 
            self.time_domain, 
            self.m, 
            traj, 
            mask
        )

    def get_collate_fn(self):
        return lambda batch: custom_collate_ODE_fn(batch, self.architecture)
    
def save_dataset(ds, path, name):

    # 1. Stack data into arrays / tensors
    values = np.stack(ds.data)  # shape [N, m]
    # To align trajectories with values (fill None with zeros):
    trajectories = np.zeros_like(values)       # shape [N, m]
    idx_sup = np.where(ds.has_trajectory)[0]
    for i, j in enumerate(idx_sup):
        trajectories[j] = ds.trajectories[j]  # Fixed: use j instead of i

    mask = ds.has_trajectory.astype(bool)  # shape [N]

    os.makedirs(path, exist_ok=True)

    name = f'{name}-date-{datetime.now().strftime("%Y-%m-%d")}.pt'
   
    torch.save({
        'values':      torch.from_numpy(values),
        'trajectories': torch.from_numpy(trajectories),
        'mask':        torch.from_numpy(mask),
        'time_domain': torch.from_numpy(ds.time_domain),
        'num_domain':  ds.num_domain,
        'num_initial': ds.num_initial,
        'end_time':    ds.end_time,
        'm':           ds.m
    }, path)

    print(f"Dataset saved to {path}")
    return path