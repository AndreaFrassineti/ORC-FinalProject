#Same idea as data_generation_p.py but for double pendulum

import casadi as cs
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cspin
from example_robot_data import load
from adam.casadi.computations import KinDynComputations
import os



def build_ocp_solver(N, nx, nu, nq, nv, f, inv_dyn, tau_min, tau_max, lbx, ubx, q_lim):
   
    opti = cs.Opti()
    
    # optimization variables
    X = opti.variable(nx, N+1)
    U = opti.variable(nu, N)
    
    # parametrize x_init
    P_init = opti.parameter(nx) 
    
    # dynamics constraints
    for i in range(N):
        opti.subject_to(X[:, i+1] == f(X[:, i], U[:, i]))
        # add torque constraint
        opti.subject_to(opti.bounded(tau_min, inv_dyn(X[:, i], U[:, i]), tau_max))
        # Physical constraints (q > q_lim)
        opti.subject_to(X[:nv, i] >= q_lim)

    # Initial constraint: X0 must be equal to the parameter P_init
    opti.subject_to(X[:, 0] == P_init)
    
    # Terminal constraint (set S): both joint velocities are zero at final time
    opti.subject_to(X[nv:, N] == 0)
    
    # # create a final state equal to 0
    # q_final = X[:nq, N]
    # v_zero  = cs.MX.zeros(nv)
    # a_zero  = cs.MX.zeros(nv)
    # x_final_static = cs.vertcat(q_final, v_zero)
    # # find the value of the torque to have a zero final state and acceleration
    # tau_hold = inv_dyn(x_final_static, a_zero)
    # # this torque must be within th torque limits
    # opti.subject_to(opti.bounded(tau_min, tau_hold, tau_max))
    
    # # Bounds on joint position and velocity
    # # we don't limit X[0], because it's the initial sampled state 
    # # X[:, 1:] take each row (pos+vel) and every column except of column 0 (state at 0)
    # opti.subject_to(opti.bounded(lbx, X[:, 1:], ubx))
    
    # Cost function
    opti.minimize(1) 
    
    
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-3, 'detect_simple_bounds': True}
    opti.solver('ipopt', opts)
    
    
    return opti, P_init, X, U


def generate_dataset_double(n_samples=5000, N=25):

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

    vMax = np.array([10.0, 10.0])
    vMin = -vMax

    # Definition of a physical constraint (mechanical constraint)
    DELTA = 0.05
    q_lim = np.array([-(np.pi+DELTA), -(0.0+DELTA)])


    

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
    
    Torque_scaling = 0.85
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

        

    # 4. Sampling loop

    print("Construction of the OCP solver")
    opti, param_x_init, X_var, U_var = build_ocp_solver(
        N, nx, nu, nq, nv, f, inv_dyn, tau_min, tau_max, lbx, ubx, q_lim)

    print(f"Dataset generation double pendulum ({n_samples} samples)...")
    data_x, data_y = [], []

    import time
    start_time = time.time()




    for i in range(n_samples):
        # Random sampling
        q_rand = np.random.uniform(-qMax, qMax)
        dq_rand = np.random.uniform(-vMax, vMax)
        x_start = np.concatenate([q_rand, dq_rand])
        
        # set the value of the parameter to the generated random state
        opti.set_value(param_x_init, x_start)

        
        
        scaling_factor = np.linspace(1.0, 0.0, N + 1)

        # 2. Posizione: Assumiamo costante (o potresti interpolare anche questa)
        # q_init ha shape (nq,), lo facciamo diventare (nq, N+1)
        q_guess = np.tile(x_start[:nq].reshape(-1, 1), (1, N + 1))

        # 3. Velocità: Scaliamo la velocità iniziale verso zero
        # dq_init * scaling_factor
        dq_guess = x_start[nq:].reshape(-1, 1) * scaling_factor.reshape(1, -1)

        # 4. Uniamo tutto
        x_guess = np.vstack((q_guess, dq_guess))

        # 5. Assegna in un colpo solo
        opti.set_initial(X_var, x_guess)
        opti.set_initial(U_var, 0.0)
        

        # 3. Solve
        try:
            opti.solve()
            label = 1
        except:
            label = 0
        
        data_x.append(x_start)
        data_y.append(label)
        
        if (i+1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {i+1}/{n_samples} | Label 1 Find: {sum(data_y)} | Time: {elapsed:.1f} s")


    # Save the dataset
    filename = f'dataset_dp_{N}.npz'
    filepath = os.path.join(folder, filename)
    np.savez(filepath, x=np.array(data_x), y=np.array(data_y))
    print(f"Dataset saved in: {filepath}")

if __name__ == "__main__":
    generate_dataset_double()