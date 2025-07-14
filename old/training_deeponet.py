import torch
import torch.nn as nn
import torch.nn.init as init
from datetime import datetime
from utils.data import MultiFunctionDatasetODE, custom_collate_ODE_fn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import utils.plotter as plotter

import torch
import torch.nn as nn
import torch.nn.init as init


class DenseNetwork(nn.Module):
    def __init__(self, layer_sizes, activations):
        super(DenseNetwork, self).__init__()
        assert len(layer_sizes) - 1 == len(activations), "Activation count must match the number of layers - 1"
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if activations[i] is not None:
                layers.append(self.get_activation(activations[i]))
        self.network = nn.Sequential(*layers)
        
        # Apply Glorot initialization
        self.apply(self.initialize_weights)

    def forward(self, x):
        return self.network(x)
    
    @staticmethod
    def get_activation(name):
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'none': nn.Identity()
        }
        if name not in activations:
            raise ValueError(f"Unsupported activation: {name}. Supported: {list(activations.keys())}")
        return activations[name]

    @staticmethod
    def initialize_weights(layer):

        if isinstance(layer, nn.Linear):
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)


class DeepONetCartesianProd(nn.Module):
    def __init__(self, branch_net, trunk_net, branch_activations, trunk_activations, input_feature=None):
        super(DeepONetCartesianProd, self).__init__()
        self.branch_net = DenseNetwork(branch_net, branch_activations)
        self.trunk_net = DenseNetwork(trunk_net, trunk_activations)


        if input_feature is not None:
            self.input_encode = input_feature
        else:
            self.input_encode = None


    def forward(self, branch_input, trunk_input):

        branch_output = self.branch_net(branch_input)  # Shape: (batch_size, branch_net[-1])
        
        if self.input_encode is not None:
            trunk_input = self.input_encode(trunk_input)
        
        trunk_output = self.trunk_net(trunk_input)    # Shape: (batch_size, trunk_net[-1])

        # Compute Cartesian product with einsum
        # Resulting shape: (batch_size, num_outputs)
        output = torch.einsum("bp,np->bn", branch_output, trunk_output)

        # Add optional bias for each output
        #output += self.b

        return output

def innitial_loss_func(model, u, t, t0, ut, batch_size, n_points):

    x_0 = model(u, t0)
    initial_loss = torch.mean((torch.ones_like(x_0) - x_0)**2)
    
    return initial_loss

def physics_loss_func(model, u, t, t0, ut, batch_size, n_points):

    x = model(u, t)

    # Physics loss
    dx = torch.zeros(batch_size, n_points, dim_x, device=t.device)
    
    # This loop is a bottleneck but i havent found a way to parallize this efficiently
    for b in range(batch_size):
        # Compute gradients for each batch independently
        dx[b] = torch.autograd.grad(x[b], t, torch.ones_like(x[b]), create_graph=True)[0]

    dx_dt = dx[:,:,0]
    # physics loss
    physics = dx_dt - (5/2)*(-x + x*ut - ut**2)
    physics_loss = torch.mean(physics**2)

    return physics_loss
    
def training(model, optimizer, dataloader, num_epochs= 1000, problem="nde", physics_loss_func=None, initial_loss_func=None, boundary_loss_func= None, plot=False):
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        for u, t, t0, ut in dataloader:

            t.requires_grad_(True)
            batch_size = u.shape[0]
            n_points = t.shape[0]

            physics_loss = physics_loss_func(model, u, t, t0, ut, batch_size, n_points) if physics_loss_func is not None else torch.zeros(1, device=t.device)
            initial_loss = initial_loss_func(model, u, t, t0, ut, batch_size, n_points) if initial_loss_func is not None else torch.zeros(1, device=t.device)
            boundary_loss = boundary_loss_func(model, u, t, t0, ut, batch_size, n_points) if boundary_loss_func is not None else torch.zeros(1, device=t.device)
            # Total loss
            loss = physics_loss + initial_loss + boundary_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, '
                    f'initial_loss: {initial_loss.item():.6f}, physics_loss: {physics_loss.item():.6f}, '
                    f'time: {datetime.now().time()}')
        
        if epoch % 100 == 0:
            if plot == True:
                plotter.GRF_test(model,m=m,lb=grf_lb,ub=grf_ub)
                plotter.linear_test(model,m=m)
                plotter.optimal_test(model,m=m)
                plotter.constant_test(model, m=m)
                plotter.polynomial_test(model, m=m)
                plotter.sine_test(model, m=m)
        
        if (epoch + 1) % 10 == 0:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            model_filename = f'model_time_[{timestamp}]_loss_[{loss.item():.4f}].pth'
            torch.save(model.state_dict(), f"trained_models/deep_o_net/{problem}/{model_filename}")
                        
        losses.append(loss)

    return model, losses

# Model Parameters
m = 100         # sensor size (branch input size)
n_hid = 200     # layer's hidden sizes
p = 200         # output size
dim_x = 1       # trunk (trunk input size)
lr = 0.0001
num_epochs = 1000

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify the MLP architecture
branch_net = [m, n_hid, n_hid, n_hid, n_hid, p]
branch_activations = ['tanh', 'tanh','tanh', 'tanh', 'none']
trunk_net = [dim_x, n_hid, n_hid, n_hid, n_hid,  p]
trunk_activations = ['tanh', 'tanh', 'tanh', 'tanh', 'none']

# Initialize model
model = DeepONetCartesianProd(branch_net, trunk_net, branch_activations, trunk_activations)
model.to(device)

#Initialize Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)


#Dataset parameters
n_functions = 100000
grf_lb = 0.05
grf_ub = 0.5
end_time = 1.0
num_domain = 200
num_initial = 20

dataset = MultiFunctionDatasetODE(
    m=m,
    n_functions=n_functions,
    function_types=['grf', 'polynomial'],
    end_time = end_time,
    num_domain = num_domain,
    num_initial = num_initial,
    grf_lb = grf_lb,
    grf_ub = grf_ub,
    project=True
)
print("===============================\nStarted generating dataset")

dataloader = DataLoader(dataset, batch_size=128, collate_fn=custom_collate_ODE_fn, shuffle=True)

print("===============================\nDataset is ready")


trained_model, lossses = training(model, optimizer, dataloader, num_epochs = num_epochs, physics_loss_func=physics_loss_func, initial_loss_func=innitial_loss_func, plot=False)