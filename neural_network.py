import torch
import torch.nn as nn
from casadi import MX, Function
from casadi import exp
import l4casadi as l4c
import os

class NeuralNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_size=32, output_size=1, activation=nn.Tanh()):
        super(NeuralNetwork, self).__init__()
        
        # architecture with 1 hidden layer composed by 32 nodes
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size)
        )
        
        # Upper bound 
        self.ub = torch.ones((output_size, 1))
        
        # weight initialization
        self.initialize_weights()

    def forward(self, x):
        self.ub = self.ub.to(x.device)
        
        if x.ndimension() == 1:
            x = x.view(1, -1)
        elif x.ndimension() == 2 and x.shape[0] == 4 and x.shape[1] != 4:
            x = x.T

        out = self.linear_stack(x) * self.ub
        return out

    def initialize_weights(self):
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias) 

    
    def create_casadi_function(self, robot_name, NN_DIR, input_size, load_weights=True):
        if load_weights:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            nn_name = os.path.join(NN_DIR, f'model_{robot_name}.pt')
            try:
                checkpoint = torch.load(nn_name, map_location=device)
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    self.load_state_dict(checkpoint['model'])
                else:
                    self.load_state_dict(checkpoint)
                print(f"Weights loaded successfully from {nn_name}")
            except Exception as e:
                print(f"Warning: Could not load weights: {e}")

        state = MX.sym("x", input_size) 
        
        self.l4c_model = l4c.L4CasADi(self,
                                      device='cuda' if torch.cuda.is_available() else 'cpu',
                                      name=f'{robot_name}_model',
                                      build_dir=os.path.join(NN_DIR, f'nn_{robot_name}'))

        logits = self.l4c_model(state)
        # Transform logits from -inf/+inf to 0/1 (Sigmoid)
        sigmoid_output = 1 / (1 + exp(-logits))
        
        self.nn_func = Function('nn_func', [state], [sigmoid_output])
        return self.nn_func