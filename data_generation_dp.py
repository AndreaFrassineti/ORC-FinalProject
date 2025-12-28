#Same idea as data_generation_p.py but for double pendulum

import casadi as cs
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cspin
from example_robot_data import load
from adam.casadi.computations import KinDynComputations
import os

def generate_dataset_double(n_samples=5000, N=10):
    # 0. Create data folder if it doesn't exist
    folder = "data"
    if not os.path.exists(folder):
        os.makedirs(folder) 

    print("--- LOADING ROBOT MODEL ---")
    robot = load("double_pendulum")
    model = robot.model
    data = model.createData()
    joints_name_list = [s for s in robot.model.names[1:]] # skip the first name because it is "universe"

    nq = model.nq
    nv = model.nv
    nx = 2 *nq
    nu = nv
    dt = 0.010 # Time step
    
    #they are zero so I need to impose them
    urdf_q_lim   = model.upperPositionLimit
    urdf_v_lim   = model.velocityLimit
    urdf_tau_lim = model.effortLimit

    print(f"URDF limits -> Pos: {urdf_q_lim[0]:.2e}, Vel: {urdf_v_lim[0]:.2f}, Tau: {urdf_tau_lim[0]:.2f}")

    # since in double_pendulum position and velocity joints limits are set to 0, I impose them manually
    qMin = np.array([-2.0*np.pi, -2.0*np.pi])
    qMax = -qMin

    vMax = np.array([8.0, 8.0])
    vMin = -vMax


    

    print("\n--- DOUBLE PENDULUM PHYSICAL INSPECTION ---")

    accumulated_length = 0.0
    total_torque = 0.0
    g = 9.81

    # I'm calculating the maximum torque when both links are horizontals and aligned, 
    # their masses are concentrated at their right extreme following the model, so this is a very conservative calculation
    for i in range(1, model.njoints):  
        inertia = model.inertias[i]
        mass = inertia.mass

        com_local = inertia.lever[2]  
        link_length = np.linalg.norm(model.jointPlacements[i].translation)

        lever_arm = accumulated_length + abs(com_local)
        tau = mass * g * lever_arm

        print(f"Link {i} | Mass: {mass:.3f} kg | Length: {link_length:.3f} m | CoM (local): {com_local:.3f} m | Torque @ base (worst-case): {tau:.3f} Nm")

        total_torque += tau
        accumulated_length += link_length

    print(f">>> Total gravity torque (joint aligned and horizontal): {total_torque:.3f} Nm")

    

    # 2. Torque (Calculated based on Physics)
    # I know that if I have tauMax >= total_torque, all the x axis (q = 0) is control inariant,
    #to complicate the things I apply a scaling factor so in some states the double pendulum isn't able to compenstate its weight
    
    Torque_scaling = 0.8 
    tauMax = np.array([total_torque * Torque_scaling, total_torque * Torque_scaling ])
    tauMin = -tauMax

   
    

    q = cs.SX.sym('q', nq)
    dq = cs.SX.sym('dq', nv)
    ddq = cs.SX.sym('u', nv)
    state = cs.vertcat(q, dq)
    rhs    = cs.vertcat(dq, ddq)
    f = cs.Function('f', [state, ddq], [state + dt * rhs])

    # create a Casadi inverse dynamics function
    kinDyn = KinDynComputations(robot.urdf, joints_name_list)
    H_b = cs.SX.eye(4)     # base configuration
    v_b = cs.SX.zeros(6)   # base velocity
    bias_forces = kinDyn.bias_force_fun()
    mass_matrix = kinDyn.mass_matrix_fun()
    # discard the first 6 elements because they are associated to the robot base
    h = bias_forces(H_b, q, v_b, dq)[6:]
    M = mass_matrix(H_b, q)[6:,6:]
    tau = M @ ddq + h
    inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])

    # Pre-compute state and torque bounds
    lbx = qMin.tolist() + vMin.tolist()   # Lower bounds: positions + velocities
    ubx = qMax.tolist() + vMax.tolist()   # Upper bounds: positions + velocities

    tau_min = tauMin.tolist()  # torque lower bounds
    tau_max = tauMax.tolist()  # torque upper bounds


    
    # 3. OCP feasibility solver

    def solve_ocp_double(x_init):
        opti = cs.Opti()
        X = opti.variable(nx, N+1)
        U = opti.variable(nu, N)

        # Dynamics constraints
        for i in range(N):
            opti.subject_to(X[:, i+1] == f(X[:, i], U[:, i]))
            
            # Add torque constraints
            opti.subject_to( opti.bounded(tau_min, inv_dyn(X[i], U[i]), tau_max))

        # Initial constraint: x_0 = x_init
        opti.subject_to(X[:, 0] == x_init)
        
        # Terminal constraint (set S): both joint velocities are zero at final time
        opti.subject_to(X[nv:, N] == 0)

        # Bounds on joint position and velocity
        # we don't limit X[0], because it's the initial sampled state 
        # X[:nq, 1:] take first 2 row (pos) and every column except of column 0 (pos 0)
        # X[nq:, 1:] take last 2 row (vel) and every column except of column 0 (vel 0)
        
        opti.subject_to(opti.bounded(-qMax, X[:nq, 1:], qMax))
        opti.subject_to(opti.bounded(-vMax, X[nq:, 1:], vMax))

        opti.minimize(1)
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-3, 'detect_simple_bounds': True}
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
        q_rand = np.random.uniform(-qMax, qMax)
        dq_rand = np.random.uniform(-vMax, vMax)
        x_start = np.concatenate([q_rand, dq_rand])
        
        label = solve_ocp_double(x_start)
        data_x.append(x_start)
        data_y.append(label)
        
        if (i+1) % 100 == 0:
            print(f"Progress: {i+1}/{n_samples} | Label 1 Find: {sum(data_y)}")

    
    # Save the dataset
    filename = f'dataset_dp_{N}.npz'
    filepath = os.path.join(folder, filename)
    np.savez(filepath, x=np.array(data_x), y=np.array(data_y))
    print(f"Dataset saved in: {filepath}")

if __name__ == "__main__":
    generate_dataset_double()