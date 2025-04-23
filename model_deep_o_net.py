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