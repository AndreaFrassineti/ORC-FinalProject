#This script aim to define the dynamics of the SIMPLE PENDULUM based on pendulum.py
# Then it generate a dataset for training a neural network to approximate the feasible set in 3 steps:
# 1) Sample random initial states x_init(q,dq)
# 2) For each initial state, solve a feasibility OCP to check if it can reach the target set S (q=0,dq=0) under input/state constraints
# 3) Store the initial state and the feasibility label (1 if feasible, 0 otherwise)

import casadi as cs
import numpy as np
import torch
from pendulum import Pendulum # Import parameters from pendulum.py
import os

def generate_dataset(n_samples=5000, N=25):

    # 0. Create data folder if it doesn't exist
    folder = "data"
    if not os.path.exists(folder):
        os.makedirs(folder) 
    # 1. Parameters Inizialization from pendulum.py environment
    env = Pendulum(nbJoint=1)
    dt = env.DT
    umax = env.umax
    vmax = env.vmax
    # Physical parameters
    m, l, g = 1.0, 1.0, 9.81 

    # 2. Symbolic Dynamic Definition
    # Used because we need symbolic expressions for the optimizer
    q = cs.SX.sym('q')
    dq = cs.SX.sym('dq')
    u = cs.SX.sym('u')
    
    # ml^2 * q_ddot + mgl*sin(q) = u - Kf*dq
    q_ddot = (u - env.Kf * dq - m * g * l * cs.sin(q)) / (m * l**2)
    x = cs.vertcat(q, dq)
    x_dot = cs.vertcat(dq, q_ddot)
    
    # Euler integration
    f = cs.Function('f', [x, u], [x + x_dot * dt])

    # 3. Funzione per risolvere l'OCP di fattibilità 
    def solve_ocp_feasibility(x_init):
        opti = cs.Opti()
        X = opti.variable(2, N+1)
        U = opti.variable(1, N)

        # Dynamics constraints
        for i in range(N):
            opti.subject_to(X[:, i+1] == f(X[:, i], U[i]))

        # Input and state constraints
        opti.subject_to(opti.bounded(-umax, U, umax))
        opti.subject_to(opti.bounded(-vmax, X[1, :], vmax))

        # Initial constraint: x_0 = x_init
        opti.subject_to(X[:, 0] == x_init)

        # Terminal constraint: x_N must be in S (velocity zero)
        opti.subject_to(X[1, N] == 0)

        # Feasibility problem: constaraint satisfaction only
        opti.minimize(1)

        # Solver options
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-3}
        opti.solver('ipopt', opts)

        try:
            opti.solve()
            return 1 # Inside the set
        except:
            return 0 # Outside the set

    # 4. Sampling loop 
    print(f"Generation of {n_samples} samples...")
    data_x = []
    data_y = []

    for i in range(n_samples):
        # Random initial state sampling
        q_start = np.random.uniform(-np.pi, np.pi)
        dq_start = np.random.uniform(-vmax, vmax)
        x_start = np.array([q_start, dq_start])
        
        label = solve_ocp_feasibility(x_start)
        
        data_x.append(x_start)
        data_y.append(label)
        
        if (i+1) % 100 == 0:
            print(f"Progress: {i+1}/{n_samples}")

    # Save the dataset
    filename = f'dataset_p_{N}.npz'
    filepath = os.path.join(folder, filename)
    np.savez(filepath, x=np.array(data_x), y=np.array(data_y))
    print(f"Dataset saved in: {filepath}")
    

if __name__ == "__main__":
    generate_dataset()