#Same idea as data_generation_p.py but for double pendulum

import casadi as cs
import numpy as np
from pendulum import Pendulum 
import os

def generate_dataset_double(n_samples=2000, N=30):
    # 0. Create data folder if it doesn't exist
    folder = "data"
    if not os.path.exists(folder):
        os.makedirs(folder) 

    # 1. Inizialization of parameters from pendulum.py
    env = Pendulum(nbJoint=2)
    dt = env.DT
    umax = env.umax # Array of input limits
    vmax = env.vmax # Velocity limits
    
    # Physical parameters
    m1, m2 = 1.0, 1.0
    l1, l2 = 1.0, 1.0
    g = 9.81

    # 2. Symbolic dynamics definition
    # x = [q1, q2, dq1, dq2]
    q = cs.SX.sym('q', 2)
    dq = cs.SX.sym('dq', 2)
    u = cs.SX.sym('u', 2)
    
    # Matrix of inertia, Coriolis and Gravity terms
    # Note: These are the standard equations of the double pendulum
    m11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * cs.cos(q[1])
    m12 = m2 * l2**2 + m2 * l1 * l2 * cs.cos(q[1])
    m22 = m2 * l2**2
    M = cs.vertcat(cs.horzcat(m11, m12), cs.horzcat(m12, m22))
    
    c1 = -m2 * l1 * l2 * (2 * dq[0] * dq[1] + dq[1]**2) * cs.sin(q[1])
    c2 = m2 * l1 * l2 * dq[0]**2 * cs.sin(q[1])
    C = cs.vertcat(c1, c2)
    
    g1 = (m1 + m2) * g * l1 * cs.sin(q[0]) + m2 * g * l2 * cs.sin(q[0] + q[1])
    g2 = m2 * g * l2 * cs.sin(q[0] + q[1])
    G = cs.vertcat(g1, g2)
    
    # ddq = M^-1 * (u - C - G - Kf*dq)
    ddq = cs.solve(M, u - C - G - env.Kf * dq)
    
    x = cs.vertcat(q, dq)
    x_dot = cs.vertcat(dq, ddq)
    f = cs.Function('f', [x, u], [x + x_dot * dt])

    # 3. OCP feasibility solver
    def solve_ocp_double(x_init):
        opti = cs.Opti()
        X = opti.variable(4, N+1)
        U = opti.variable(2, N)

        for i in range(N):
            opti.subject_to(X[:, i+1] == f(X[:, i], U[:, i]))

        # Constraints: u in [-umax, umax], dq in [-vmax, vmax]
        opti.subject_to(opti.bounded(-umax, U, umax))
        opti.subject_to(opti.bounded(-vmax, X[2:, :], vmax))
        
        opti.subject_to(X[:, 0] == x_init)
        
        # Set S: both velocities are zero at final time
        opti.subject_to(X[2:, N] == 0)

        opti.minimize(1)
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-3}
        opti.solver('ipopt', opts)

        try:
            opti.solve()
            return 1
        except:
            return 0

    # 4. Sampling loop ( q1, q2 in [-pi, pi], dq1, dq2 in [-vmax, vmax])
    print(f"Dataset generation double pendulum ({n_samples} samples)...")
    data_x, data_y = [], []

    for i in range(n_samples):
        # Sampling q1, q2 in [-pi, pi] and dq1, dq2 in [-vmax, vmax]
        q_rand = np.random.uniform(-np.pi, np.pi, 2)
        dq_rand = np.random.uniform(-vmax, vmax, 2)
        x_start = np.concatenate([q_rand, dq_rand])
        
        label = solve_ocp_double(x_start)
        data_x.append(x_start)
        data_y.append(label)
        
        if (i+1) % 100 == 0:
            print(f"Progress: {i+1}/{n_samples} | Label 1 Find: {sum(data_y)}")

    np.savez('dataset_double_pendulum.pt', x=np.array(data_x), y=np.array(data_y))
    print("Dataset saved!")
    filepath = os.path.join(folder, 'dataset_double_pendulum.pt')
    np.savez(filepath, x=np.array(data_x), y=np.array(data_y))
    print(f"Dataset saved in: {filepath}")

if __name__ == "__main__":
    generate_dataset_double()