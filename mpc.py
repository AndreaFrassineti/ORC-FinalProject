import casadi as ca
import numpy as np
import torch
from neural_network import NeuralNetwork
import l4casadi as l4c
from pendulum import Pendulum

def run_mpc_control(model_path, n_steps=20):
    # 1. Initialize the system and neural network
    pendulum = Pendulum(open_viewer=False)
    input_dim = 2 # q, dq

    # Load the neural network model defined in your file
    nn_model = NeuralNetwork(input_size=input_dim, hidden_size=32, output_size=1)

    # Load weights from .pt file
    checkpoint = torch.load(model_path)
    nn_model.load_state_dict(checkpoint['model'])
    nn_model.eval()

    # 2. Creation of the L4CasADi function
    # This transforms the PyTorch model into a symbolic CasADi function
    l4c_model = l4c.L4CasADi(nn_model, device='cpu')

    # 3. MPC formulation with CasADi Opti
    opti = ca.Opti()

    # Decision variables
    x = opti.variable(2, n_steps + 1) # States: [q, dq]
    u = opti.variable(1, n_steps)     # Motor torque
    
    # Parameter for the current state (will be updated each cycle)
    x_init = opti.parameter(2)
    
    # Cost function (LQR-like)
    cost = 0
    Q = np.diag([10, 0.1])
    R = 0.01
    
    for k in range(n_steps):
        # Minimize the position error and control effort
        cost += ca.mtimes([x[:, k].T, Q, x[:, k]]) + R * u[k]**2
        
        # Dynamics (simplified Euler integration or RK4)
        x_next = pendulum.dynamics_step_casadi(x[:, k], u[k]) # Method to implement in pendulum.py
        opti.subject_to(x[:, k+1] == x_next)
        
        # --- VIABILITY CONSTRAINT ---
        # We use the neural network to tell the MPC: "Stay in the green!"
        # Based on the graph learned by the NN. If the NN output >= 0.5, it's safe., otherwise the optimal control steers away from unsafe regions.
        opti.subject_to(l4c_model(x[:, k+1]) >= 0.5)
        
        # Motor physical limits
        opti.subject_to(opti.bounded(-pendulum.umax, u[k], pendulum.umax))

    opti.minimize(cost)
    opti.subject_to(x[:, 0] == x_init)
    
    # Solver options
    opts = {"ipopt.print_level": 0, "print_time": False}
    opti.solver('ipopt', opts)
    
    return opti, x_init, u