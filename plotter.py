import matplotlib.pyplot as plt
import torch
import numpy as np
import math
from data import generate_gaussian_random_field_1d
import models
from scipy.interpolate import griddata


def feldbaum_euler(t,x0,u):
    y = np.zeros_like(t)
    n = len(y)
    y[0] = x0
    
    for i in range(0,n-1):
        dt = t[i+1]-t[i]
        y[i+1] = y[i] + (-y[i]+u[i])*dt

    return y

def finaltime_euler(t,x0,u):
    y = np.zeros_like(t)
    n = len(y)
    y[0] = x0
    
    for i in range(0,n-1):
        dt = t[i+1]-t[i]
        y[i+1] = y[i] + (5/2)*(-y[i] + y[i]*u[i] - u[i]**2)*dt

    return y

def optimal_x(t):
    sq2 = math.sqrt(2)
    numerator = sq2*math.cosh(sq2*(t-1.0))-math.sinh(sq2*(t-1))
    denominator = sq2*math.cosh(sq2) + math.sinh(sq2)
    return numerator/denominator

def optimal_lambda(t):
    sq2 = math.sqrt(2)
    numerator = -math.sinh(sq2*(t-1.0))
    denominator = sq2*math.cosh(sq2)+math.sinh(sq2)
    return numerator/denominator

def optimal_u(t):
    return -optimal_lambda(t)

def optimal_x_finaltime(t):
    a = 1+3*math.exp((5*t)/2)
    b = math.exp(-5)+6+9*math.exp(5)
    x = 4/a
    return x 

def optimal_lambda_finaltime(t):
    a = 1+3*math.exp((5*t)/2)
    b = math.exp(-5)+6+9*math.exp(5)
    return -math.exp(2*math.log(a)-(5.0/2.0)*t)/b


def optimal_u_finaltime(t):
    return optimal_x(t)/2

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


def linear_test(model,m=200,lb=-2,ub=2):
    model.eval()
    
    # Plot realisations of v(t)
    # Number of cases to plot
    n_cases = 9
    n_rows, n_cols = 3, 3

    # Generate random features for each case
    n = n_cases

    # Create the subplot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    # Define the domain and parameters
    grid_size = 512  # Number of points

    import random
    # Loop through each case and plot in the subplot
    for i, ax in enumerate(axes.flatten()):
        
        lb_ = random.uniform(lb, 0)
        ub_ = random.uniform(0, ub)
        
        # Generate the Gaussian random field
        t_np = np.linspace(0,1,m)
        u_np = np.linspace(lb_,ub_,m)
        
        if np.random.rand() < 0.5:  # Generate a random number in [0, 1) and check if it's less than 0.5
            u_np = u_np[::-1].copy()  # Reverse the array
        
        t = torch.linspace(0,1,m).unsqueeze(-1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        u = torch.from_numpy(u_np).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        u = u.unsqueeze(0)#.repeat(m,1)

        y = model(u,t).squeeze(-1)
        t_np = t.squeeze(1).detach().cpu().numpy()

        y_analytical = feldbaum_euler(t_np,x0=1.0,u=u_np)

        # Plot the results in the current subplot
        
        ax.plot(t_np,y.detach().cpu().numpy().flatten(),label="DeepONet", color = "darkorange",linewidth=2.0)
        ax.plot(t_np,y_analytical,label="analytical", color="black",linewidth=0.75,linestyle="--")
        ax.plot(t_np,u_np,label="u(t)",color="gray",linewidth=1.0)

        # Configure the subplot
        ax.set_title(f"Case {i + 1}")
        ax.legend()
        ax.grid(True)

    plt.savefig("feldbaum_linear.png")
    plt.close()
    
    return


def GRF_test(model,m=200,lb=0.02, ub=0.5):
    model.eval()
    
    n_cases = 9
    n_rows, n_cols = 3, 3

    # Generate random features for each case
    n = n_cases

    # Create the subplot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    # Define the domain and parameters
    grid_size = 512  # Number of points

    import random
    # Loop through each case and plot in the subplot
    
    # Loop through each case and plot in the subplot
    for i, ax in enumerate(axes.flatten()):
        
        length_scale = random.uniform(lb,ub)
        
        # Generate the Gaussian random field
        field = generate_gaussian_random_field_1d(grid_size, length_scale)
        t_new = np.linspace(0,1,m)
        t_old = np.linspace(0,1,grid_size)
        grf = np.interp(t_new,t_old,field)

        t = torch.linspace(0,1,grid_size).unsqueeze(-1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        u = torch.from_numpy(grf).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        #u = u.unsqueeze(0).repeat(grid_size,1)
        u = u.unsqueeze(0)

        #y = model(u,t).squeeze(-1)
        y = model(u,t)
        t_np = t.squeeze(1).detach().cpu().numpy().flatten()

        y_analytical = feldbaum_euler(t_np,x0=1.0,u=field)

        # Plot the results in the current subplot
        
        ax.plot(t_np,y.detach().cpu().numpy().flatten(),label="DeepONet", color = "darkorange",linewidth=2.0)
        ax.plot(t_np,y_analytical,label="analytical", color="black",linewidth=0.75,linestyle="--")
        ax.plot(t_np,field,label="u(t)",color="gray",linewidth=1.0)

        # Configure the subplotd
        ax.set_title(f"Case {i + 1}")
        ax.legend()
        ax.grid(True)

    plt.savefig("feldbaum_grf.png")
    plt.close()
    return
    


def optimal_test(model,m=200):
    model.eval()

    t_np = np.linspace(0,1,m)
    t = torch.linspace(0,1,m).unsqueeze(-1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    u_opt = np.zeros(m)
    y_opt = np.zeros(m)
    for i in range(0,m):
        u_opt[i] = optimal_u(t_np[i])
        y_opt[i] = optimal_x(t_np[i])
        
    u_opt_torch = torch.from_numpy(u_opt).float()
    #u_opt_torch = u_opt_torch.unsqueeze(0).repeat(m,1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    u_opt_torch = u_opt_torch.unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    y = model(u_opt_torch,t).squeeze(-1)

    plt.figure()
    plt.plot(t_np,y.detach().cpu().numpy().flatten(),label="DeepONet", color = "darkorange",linewidth=2.0)
    plt.plot(t_np,y_opt,label="analytical", color="black",linewidth=0.75,linestyle="--")
    plt.plot(t_np,u_opt,label="u(t)",color="gray",linewidth=1.0)

    plt.savefig("feldbaum_optimal.png")
    plt.close()
    return

def constant_test(model,m=200,lb=-3,ub=3):
    model.eval()
    
    # Plot realisations of v(t)
    # Number of cases to plot
    n_cases = 9
    n_rows, n_cols = 3, 3

    # Generate random features for each case
    n = n_cases

    # Create the subplot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    # Define the domain and parameters
    grid_size = 512  # Number of points

    import random
    # Loop through each case and plot in the subplot
    for i, ax in enumerate(axes.flatten()):
        
        const = random.uniform(lb, ub)

        # Generate the Gaussian random field
        t_np = np.linspace(0,1,m)
        u_np = np.ones_like(t_np)*const
        
        t = torch.linspace(0,1,m).unsqueeze(-1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        u = torch.from_numpy(u_np).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        u = u.unsqueeze(0)#.repeat(m,1)

        y = model(u,t).squeeze(-1)
        t_np = t.squeeze(1).detach().cpu().numpy()

        y_analytical = feldbaum_euler(t_np,x0=1.0,u=u_np)

        # Plot the results in the current subplot
        
        ax.plot(t_np,y.detach().cpu().numpy().flatten(),label="DeepONet", color = "darkorange",linewidth=2.0)
        ax.plot(t_np,y_analytical,label="analytical", color="black",linewidth=0.75,linestyle="--")
        ax.plot(t_np,u_np,label="u(t)",color="gray",linewidth=1.0)

        # Configure the subplot
        ax.set_title(f"Case {i + 1}")
        ax.legend()
        ax.grid(True)

    plt.savefig("feldbaum_constant.png")
    plt.close()
    
    return

def polynomial_test(model,m=200,lb=-3,ub=3,name="feldbaum_polynomial.png"):
    model.eval()
    model_dtype = next(model.parameters()).dtype
    # Plot realisations of v(t)
    # Number of cases to plot
    n_cases = 9
    n_rows, n_cols = 3, 3

    # Generate random features for each case
    n = n_cases

    # Create the subplot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    # Define the domain and parameters
    grid_size = 512  # Number of points

    import random
    # Loop through each case and plot in the subplot
    for i, ax in enumerate(axes.flatten()):
        

        # Generate the Gaussian random field
        t_np = np.linspace(0,1,m)
        coefficients = np.random.uniform(lb, ub, size=np.random.randint(3, 8))
        u_np = np.polyval(coefficients, t_np)
        #u_np = project_to_range(u_np)
        
        t = torch.linspace(0,1,m).unsqueeze(-1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(dtype=model_dtype)
        u = torch.from_numpy(u_np).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(dtype=model_dtype)
        u = u.unsqueeze(0)#.repeat(m,1)

        y = model(u,t)
        t_np = t.squeeze(1).detach().cpu().numpy()

        y_analytical = feldbaum_euler(t_np,x0=1.0,u=u_np)

        # Plot the results in the current subplot
        
        ax.plot(t_np,y.detach().cpu().numpy().flatten(),label="DeepONet", color = "darkorange",linewidth=2.0)
        ax.plot(t_np,y_analytical,label="analytical", color="black",linewidth=0.75,linestyle="--")
        ax.plot(t_np,u_np,label="u(t)",color="gray",linewidth=1.0)

        # Configure the subplot
        ax.set_title(f"Case {i + 1}")
        ax.legend()
        ax.grid(True)

    plt.savefig(name)
    plt.close()
    
    return

def sine_test(model,m=200,lb=-3,ub=3,name="feldbaum_sine.png"):
    model.eval()
    model_dtype = next(model.parameters()).dtype
    # Plot realisations of v(t)
    # Number of cases to plot
    n_cases = 9
    n_rows, n_cols = 3, 3

    # Generate random features for each case
    n = n_cases

    # Create the subplot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    # Define the domain and parameters
    grid_size = 512  # Number of points

    import random
    # Loop through each case and plot in the subplot
    for i, ax in enumerate(axes.flatten()):
        

        # Generate the Gaussian random field
        t_np = np.linspace(0,1,m)
        frequency = np.random.uniform(0.1, 10)  # Random frequency
        amplitude = np.random.uniform(0.5, 2)  # Random amplitude
        phase = np.random.uniform(0, 2 * np.pi)  # Random phase
        u_np = amplitude * np.sin(2 * np.pi * frequency * t_np + phase)
        #u_np = project_to_range(u_np)
        
        t = torch.linspace(0,1,m).unsqueeze(-1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(dtype=model_dtype)
        u = torch.from_numpy(u_np).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(dtype=model_dtype)
        u = u.unsqueeze(0)#.repeat(m,1)

        y = model(u,t)
        t_np = t.squeeze(1).detach().cpu().numpy()

        y_analytical = feldbaum_euler(t_np,x0=1.0,u=u_np)

        # Plot the results in the current subplot
        
        ax.plot(t_np,y.detach().cpu().numpy().flatten(),label="DeepONet", color = "darkorange",linewidth=2.0)
        ax.plot(t_np,y_analytical,label="analytical", color="black",linewidth=0.75,linestyle="--")
        ax.plot(t_np,u_np,label="u(t)",color="gray",linewidth=1.0)

        # Configure the subplot
        ax.set_title(f"Case {i + 1}")
        ax.legend()
        ax.grid(True)

    plt.savefig(name)
    plt.close()
    
    return

def GRF_test_finaltime(model,m=100,lb=0.02, ub=0.3,name="finaltime_grf.png"):
    model.eval()
    
    model_dtype = next(model.parameters()).dtype
    
    n_cases = 9
    n_rows, n_cols = 3, 3

    # Generate random features for each case
    n = n_cases

    # Create the subplot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    # Define the domain and parameters
    grid_size = 512  # Number of points

    import random
    # Loop through each case and plot in the subplot
    
    # Loop through each case and plot in the subplot
    for i, ax in enumerate(axes.flatten()):
        
        length_scale = random.uniform(lb,ub)
        
        # Generate the Gaussian random field
        field = generate_gaussian_random_field_1d(grid_size, length_scale,end_time=2)
        t_new = np.linspace(0,2,m)
        t_old = np.linspace(0,2,grid_size)
        grf = np.interp(t_new,t_old,field)
        #grf = project_to_range(grf)
        #field = project_to_range(field)

        t = torch.linspace(0,2,m).to(dtype=model_dtype).unsqueeze(-1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        u = torch.from_numpy(grf).to(dtype=model_dtype).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        u = u.unsqueeze(0)

        y = model(u,t)
        t_np = t.squeeze(1).detach().cpu().numpy()

        y_analytical = finaltime_euler(t_np,x0=1.0,u=grf)

        # Plot the results in the current subplot
        y_np = y.detach().cpu().numpy().flatten()
        loss = np.mean((y_analytical-y_np)**2)
        ax.plot(t_np,y_np,label="DeepONet", color = "darkorange",linewidth=2.0)
        ax.plot(t_np,y_analytical,label="analytical", color="black",linewidth=0.75,linestyle="--")
        ax.plot(t_np,grf,label="u(t)",color="gray",linewidth=1.0)

        # Configure the subplotd
        ax.set_title(f"Case {i + 1}, Loss: {loss:.6f}")
        ax.legend()
        ax.grid(True)

    plt.savefig(name)
    plt.close()
    return

def optimal_test_finaltime(model,m=200,name="finaltime_optimal.png"):
    model.eval()
    model_dtype = next(model.parameters()).dtype

    t_np = np.linspace(0,2,m)
    t = torch.linspace(0,2,m).unsqueeze(-1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(dtype=model_dtype)
    
    u_opt = np.zeros(m)
    y_opt = np.zeros(m)
    for i in range(0,m):
        u_opt[i] = optimal_u_finaltime(t_np[i])
        y_opt[i] = optimal_x_finaltime(t_np[i])
        
    u_opt_torch = torch.from_numpy(u_opt).to(dtype=model_dtype)
    #u_opt_torch = u_opt_torch.unsqueeze(0).repeat(m,1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    u_opt_torch = u_opt_torch.unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    y = model(u_opt_torch,t).squeeze(-1)

    plt.figure()
    plt.plot(t_np,y.detach().cpu().numpy().flatten(),label="DeepONet", color = "darkorange",linewidth=2.0)
    plt.plot(t_np,y_opt,label="analytical", color="black",linewidth=0.75,linestyle="--")
    plt.plot(t_np,u_opt,label="u(t)",color="gray",linewidth=1.0)

    plt.savefig(name)
    plt.close()
    return


def linear_test_finaltime(model,m=200,lb=-2,ub=2,name="finaltime_linear.png"):
    model.eval()
    model_dtype = next(model.parameters()).dtype
    
    # Plot realisations of v(t)
    # Number of cases to plot
    n_cases = 9
    n_rows, n_cols = 3, 3

    # Generate random features for each case
    n = n_cases

    # Create the subplot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    # Define the domain and parameters
    grid_size = 512  # Number of points

    import random
    # Loop through each case and plot in the subplot
    for i, ax in enumerate(axes.flatten()):
        
        lb_ = random.uniform(lb, 0)
        ub_ = random.uniform(0, ub)
        
        # Generate the Gaussian random field
        t_np = np.linspace(0,2,m)
        u_np = np.linspace(lb_,ub_,m)
        #u_np = project_to_range(u_np)
        
        if np.random.rand() < 0.5:  # Generate a random number in [0, 1) and check if it's less than 0.5
            u_np = u_np[::-1].copy()  # Reverse the array
        
        t = torch.linspace(0,2,m).unsqueeze(-1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(dtype=model_dtype)
        u = torch.from_numpy(u_np).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(dtype=model_dtype)
        u = u.unsqueeze(0)#.repeat(m,1)
        

        y = model(u,t)
        t_np = t.squeeze(1).detach().cpu().numpy()

        y_analytical = finaltime_euler(t_np,x0=1.0,u=u_np)

        # Plot the results in the current subplot
        
        ax.plot(t_np,y.detach().cpu().numpy().flatten(),label="DeepONet", color = "darkorange",linewidth=2.0)
        ax.plot(t_np,y_analytical,label="analytical", color="black",linewidth=0.75,linestyle="--")
        ax.plot(t_np,u_np,label="u(t)",color="gray",linewidth=1.0)

        # Configure the subplot
        ax.set_title(f"Case {i + 1}")
        ax.legend()
        ax.grid(True)

    plt.savefig(name)
    plt.close()
    
    return

def constant_test_finaltime(model,m=200,lb=-1,ub=1,name="finaltime_constant.png"):
    model.eval()
    model_dtype = next(model.parameters()).dtype
    # Plot realisations of v(t)
    # Number of cases to plot
    n_cases = 9
    n_rows, n_cols = 3, 3

    # Generate random features for each case
    n = n_cases

    # Create the subplot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    # Define the domain and parameters
    grid_size = 512  # Number of points

    import random
    # Loop through each case and plot in the subplot
    for i, ax in enumerate(axes.flatten()):
        
        const = random.uniform(lb, ub)

        # Generate the Gaussian random field
        t_np = np.linspace(0,2,m)
        u_np = np.ones_like(t_np)*const
        #u_np = project_to_range(u_np)
        
        t = torch.linspace(0,2,m).unsqueeze(-1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(dtype=model_dtype)
        u = torch.from_numpy(u_np).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(dtype=model_dtype)
        u = u.unsqueeze(0)#.repeat(m,1)
        

        y = model(u,t)
        t_np = t.squeeze(1).detach().cpu().numpy()

        y_analytical = finaltime_euler(t_np,x0=1.0,u=u_np)

        # Plot the results in the current subplot
        
        ax.plot(t_np,y.detach().cpu().numpy().flatten(),label="DeepONet", color = "darkorange",linewidth=2.0)
        ax.plot(t_np,y_analytical,label="analytical", color="black",linewidth=0.75,linestyle="--")
        ax.plot(t_np,u_np,label="u(t)",color="gray",linewidth=1.0)

        # Configure the subplot
        ax.set_title(f"Case {i + 1}")
        ax.legend()
        ax.grid(True)

    plt.savefig(name)
    plt.close()
    
    return

def polynomial_test_finaltime(model,m=200,lb=-3,ub=3,name="finaltime_polynomial.png"):
    model.eval()
    model_dtype = next(model.parameters()).dtype
    # Plot realisations of v(t)
    # Number of cases to plot
    n_cases = 9
    n_rows, n_cols = 3, 3

    # Generate random features for each case
    n = n_cases

    # Create the subplot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    # Define the domain and parameters
    grid_size = 512  # Number of points

    import random
    # Loop through each case and plot in the subplot
    for i, ax in enumerate(axes.flatten()):
        

        # Generate the Gaussian random field
        t_np = np.linspace(0,2,m)
        coefficients = np.random.uniform(lb, ub, size=np.random.randint(2, 5))
        u_np = np.polyval(coefficients, t_np)
        #u_np = project_to_range(u_np)
        
        t = torch.linspace(0,2,m).unsqueeze(-1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(dtype=model_dtype)
        u = torch.from_numpy(u_np).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(dtype=model_dtype)
        u = u.unsqueeze(0)#.repeat(m,1)

        y = model(u,t)
        t_np = t.squeeze(1).detach().cpu().numpy()

        y_analytical = finaltime_euler(t_np,x0=1.0,u=u_np)

        # Plot the results in the current subplot
        
        ax.plot(t_np,y.detach().cpu().numpy().flatten(),label="DeepONet", color = "darkorange",linewidth=2.0)
        ax.plot(t_np,y_analytical,label="analytical", color="black",linewidth=0.75,linestyle="--")
        ax.plot(t_np,u_np,label="u(t)",color="gray",linewidth=1.0)

        # Configure the subplot
        ax.set_title(f"Case {i + 1}")
        ax.legend()
        ax.grid(True)

    plt.savefig(name)
    plt.close()
    
    return
def sine_test_finaltime(model,m=200,lb=-3,ub=3,name="finaltime_polynomial.png"):
    model.eval()
    model_dtype = next(model.parameters()).dtype
    # Plot realisations of v(t)
    # Number of cases to plot
    n_cases = 9
    n_rows, n_cols = 3, 3

    # Generate random features for each case
    n = n_cases

    # Create the subplot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    # Define the domain and parameters
    grid_size = 512  # Number of points

    import random
    # Loop through each case and plot in the subplot
    for i, ax in enumerate(axes.flatten()):
        


        # Generate the Gaussian random field
        t_np = np.linspace(0,2,m)
        frequency = np.random.uniform(1, 5)  # Random frequency
        amplitude = np.random.uniform(0.5, 1)  # Random amplitude
        phase = np.random.uniform(0, 2 * np.pi)  # Random phase
        u_np = amplitude * np.sin(2 * np.pi * frequency * t_np + phase)
        #u_np = project_to_range(u_np)
        
        t = torch.linspace(0,2,m).unsqueeze(-1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(dtype=model_dtype)
        u = torch.from_numpy(u_np).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).to(dtype=model_dtype)
        u = u.unsqueeze(0)#.repeat(m,1)

        y = model(u,t)
        t_np = t.squeeze(1).detach().cpu().numpy()

        y_analytical = finaltime_euler(t_np,x0=1.0,u=u_np)

        # Plot the results in the current subplot
        
        ax.plot(t_np,y.detach().cpu().numpy().flatten(),label="DeepONet", color = "darkorange",linewidth=2.0)
        ax.plot(t_np,y_analytical,label="analytical", color="black",linewidth=0.75,linestyle="--")
        ax.plot(t_np,u_np,label="u(t)",color="gray",linewidth=1.0)

        # Configure the subplot
        ax.set_title(f"Case {i + 1}")
        ax.legend()
        ax.grid(True)

    plt.savefig(name)
    plt.close()
    
    return

def test_diffusion(model,m=50,length_scale=0.3):
    model.eval()
    
    v_in = generate_gaussian_random_field_1d(grid_size=m,length_scale=length_scale)
    
    x, t, y_true = solve_ADR(
        0,
        1,
        0,
        1,
        lambda x: 0.01 * np.ones_like(x),
        lambda x: np.zeros_like(x),
        lambda u: 0.01 * u**2,
        lambda u: 0.02 * u,
        lambda x, t: np.tile(v_in[:, None], (1, len(t))),
        lambda x: np.zeros_like(x),
        m,
        m,
    )
    y_true = y_true.T
    
    plt.figure()
    plt.subplot(2,2,1)
    img = plt.imshow(y_true)
    plt.colorbar(img, fraction=0.046, pad=0.04)
    plt.subplot(2,2,3)
    plt.plot(v_in.flatten(),label="u_in", color="black")
    plt.legend()
    
    
    x = np.linspace(0, 1, num=m)
    t = np.linspace(0, 1, num=m)
    xv, tv = np.meshgrid(x, t)
    x_trunk = np.vstack((np.ravel(tv), np.ravel(xv))).T
    x_tensor = torch.from_numpy(x_trunk).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    u_tensor = torch.from_numpy(v_in).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    M = x_trunk.shape[0]
    y = model(u_tensor.unsqueeze(0),x_tensor).squeeze(-1)
    y = y.reshape(m,m)
    
    plt.subplot(2,2,2)
    img = plt.imshow(y.detach().cpu().numpy())
    plt.colorbar(img, fraction=0.046, pad=0.04)
    plt.subplot(2,2,4)
    plt.plot(v_in.flatten(),label="u_in", color="black")
    plt.legend()
    
    plt.savefig("diffusion.png")
    plt.close()
    return

def test_burgers(model,m=50,length_scale=0.3):
    model.eval()

    grid_size = 512
    t_new = np.linspace(0,1,m)
    t_old = np.linspace(0,1,grid_size)
    
    
    grf = generate_periodic_gaussian_random_field_1d(grid_size=grid_size,length_scale=length_scale)
    #grf = grf/np.max(np.abs(grf))
    grf = project_to_range(grf)
    v_in = np.interp(t_new,t_old,grf)
    
    plt.figure()
    
    x = np.linspace(0, 1, num=m)
    t = np.linspace(0, 1, num=m)
    xv, tv = np.meshgrid(x, t)
    x_trunk = np.vstack((np.ravel(tv), np.ravel(xv))).T
    x_tensor = torch.from_numpy(x_trunk).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    u_tensor = torch.from_numpy(v_in).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    y = model(u_tensor.unsqueeze(0),x_tensor).squeeze(-1)
    y = y.reshape(m,m)
    
    plt.subplot(1,2,1)
    img = plt.imshow(y.detach().cpu().numpy())
    plt.colorbar(img, fraction=0.046, pad=0.04)

    plt.subplot(1,2,2)
    x0 = torch.zeros(m,2).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    x0[:,1] = torch.linspace(0,1,m)
    y0 = model(u_tensor.unsqueeze(0),x0).squeeze(0)

    plt.title(f"initial_condition_loss: {torch.mean((u_tensor-y0)**2).item()}")
    plt.plot(y0.detach().cpu().flatten(),label="DeepONet", color="darkorange")
    plt.plot(v_in.flatten(),label="u_in", color="black",linewidth=1)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("burgers.png")
    plt.close()

def test_poisson(model,m=20,length_scale=0.5):
    model.eval()

    grid_size = 512
    end_x = 1.0
    end_y = 1.0

    
    grf = generate_gaussian_random_field_2d(grid_size, length_scale,end_x=end_x,end_y=end_y)
    grf = grf/np.max(np.abs(grf))
    grf_values = grf.ravel()

    grid_x = np.linspace(0, end_x, grid_size)
    grid_y = np.linspace(0, end_y, grid_size)
    X_fine, Y_fine = np.meshgrid(grid_x, grid_y)
    points = np.array([X_fine.ravel(), Y_fine.ravel()]).T

    m_x = np.linspace(0, end_x, m)
    m_y = np.linspace(0, end_y, m)
    m_grid_x, m_grid_y = np.meshgrid(m_x,m_y) 
            
    u_np = griddata(points, grf_values, (m_grid_x, m_grid_y), method='linear')
    u_np_flat = u_np.flatten()

    
    plt.figure()
    
    x = np.linspace(0, 1, num=m)
    y = np.linspace(0, 1, num=m)
    xgrid, ygrid  = np.meshgrid(x, y)
    x_trunk = np.vstack((np.ravel(xgrid), np.ravel(ygrid))).T
    x_tensor = torch.from_numpy(x_trunk).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    u_tensor = torch.from_numpy(u_np_flat).float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    y = model(u_tensor.unsqueeze(0),x_tensor).squeeze(0)
    y = y.reshape(m,m)
    
    plt.subplot(1,2,1)
    img = plt.imshow(y.detach().cpu().numpy())
    plt.colorbar(img, fraction=0.046, pad=0.04)

    plt.subplot(1,2,2)
    img = plt.imshow(u_np)
    plt.colorbar(img, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("poisson.png")
    plt.close()