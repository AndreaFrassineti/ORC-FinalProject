import os
import sys
import time
import numpy as np
import casadi as cs
import pinocchio as pin
from pinocchio import casadi as cspin
from example_robot_data import load
from adam.casadi.computations import KinDynComputations
import matplotlib.pyplot as plt
from termcolor import colored

try:
    from train import NeuralNetwork
    from utils.robot_simulator import RobotSimulator
    from utils.robot_wrapper import RobotWrapper
    import conf_doublep as conf_doublep 
except ImportError:
    print("ERROR: Ensure train.py and utils are accessible.")
    sys.exit(1)

ROBOT_NAME = "double_pendulum"
NN_MODEL_NAME = "dp_100" # Name of the model
NN_DIR = "models"       # Folder where the model is stored

# MPC Parameters
N_MPC = 30              # MPC Horizon 
DT = 0.010              # Integration step, same as dataset generation
PROB_THRESHOLD = 0.5    # Neural Network safety threshold
USE_NEURAL_NETWORK = 1  # if is set to 1 use the terminal constraint, otherwise solve the mpc without the nn
SEED = None  # set the seed to reply the same random state for every N. If it is none the state is complitely random

# Cost weights
W_POS = 1e3   # Position
W_VEL = 1e-1  # Velocity
W_ACC = 1e-4  # Acceleration

# Solver construction
def build_mpc_solver(N, nx, nu, nq, nv, f, inv_dyn, tau_min, tau_max, lbx, ubx, q_lim, nn_constraint_fun, target_q, use_nn):
    
    opti = cs.Opti()
    
    # Decision variables
    X = opti.variable(nx, N+1) 
    U = opti.variable(nu, N) 
    
    P_init = opti.parameter(nx)
    P_target = opti.parameter(nq) 
    
    # Initial State Constraint
    opti.subject_to(X[:, 0] == P_init)
    
    # Dynamics & Physical Constraints Loop
    for i in range(N):
        opti.subject_to(X[:, i+1] == f(X[:, i], U[:, i]))
        
        tau_val = inv_dyn(X[:, i], U[:, i])
        opti.subject_to(opti.bounded(tau_min, tau_val, tau_max))
        
        # Physical constraints (q > q_lim)
        opti.subject_to(X[:nq, i+1] >= q_lim)

    # Bounds on joint position and velocity
    opti.subject_to(opti.bounded(lbx, X[:, 1:], ubx))
    
    # TERMINAL CONSTRAINT (Neural Network)
    # Applies the function learned by the neural network
    # If > 0.5, the point belongs to the Backward Reachable Set (BRS)
    if use_nn:
        final_state = X[:, N]
        opti.subject_to(nn_constraint_fun(final_state) >= PROB_THRESHOLD)
    
    # --- COST FUNCTION ---
    cost = 0
    for i in range(N):
        # Position error
        e_pos = X[:nq, i] - P_target
        # Velocity
        v_curr = X[nq:, i]
        # Acceleration (proxy for effort)
        acc = U[:, i]
        
        cost += W_POS * cs.sumsqr(e_pos)
        cost += W_VEL * cs.sumsqr(v_curr)
        cost += W_ACC * cs.sumsqr(acc)
    
    # Terminal cost (Optional)
    # cost += 10 * W_VEL * cs.dot(X[nq:, N], X[nq:, N]) # Brake at the end
    
    opti.minimize(cost)
    
    # --- SOLVER SETTINGS (IPOPT) ---
    opts = {
        'ipopt.print_level': 0, 
        'print_time': 0, 
        'ipopt.tol': 1e-4, 
        'ipopt.max_iter': 100,
        'detect_simple_bounds': True,
        # 'ipopt.hessian_approximation': 'limited-memory' # Uncomment if slow
    }
    opti.solver('ipopt', opts)
    
    return opti, P_init, P_target, X, U

# --- MAIN ---
def main():
    # 1. Load Robot & Model
    print("--- LOADING ROBOT ---")
    robot = load(ROBOT_NAME)
    model = robot.model
    data = model.createData()
    joints_name_list = [s for s in robot.model.names[1:]]

    nq = model.nq
    nv = model.nv
    nx = 2 * nq
    nu = nv
    
    # Physical inspection of the model to find torque bounds
    accumulated_length = 0.0
    total_torque = 0.0
    g = 9.81
    
    for i in range(1, model.njoints):  
        inertia = model.inertias[i]
        mass = inertia.mass
        com_local = inertia.lever[2]  
        link_length = np.linalg.norm(model.jointPlacements[i].translation)
        lever_arm = accumulated_length + abs(com_local)
        tau_g = mass * g * lever_arm
        total_torque += tau_g
        accumulated_length += link_length
    
    Torque_scaling = 1.2
    
    tau_lim_abs = total_torque * Torque_scaling
    tau_max_list = [tau_lim_abs] * nu
    tau_min_list = [-tau_lim_abs] * nu
    
    vMax = np.array([10.0, 10.0])
    vMin = -vMax
    qMin = np.array([-2.0*np.pi, -2.0*np.pi])
    qMax = -qMin
    
    lbx = qMin.tolist() + vMin.tolist()
    ubx = qMax.tolist() + vMax.tolist()
    
    # Wall limits
    DELTA = 0.05
    q_lim = np.array([-(np.pi+DELTA), -(0.0+DELTA)])
    
    # Target (set near wall limits to test behavior)
    q_des_val = np.array([-np.pi + 0.2, 0.0]) 

    # CasADi Dynamics Functions
    q_sym = cs.SX.sym('q', nq)
    dq_sym = cs.SX.sym('dq', nv)
    ddq_sym = cs.SX.sym('ddq', nu)
    state_sym = cs.vertcat(q_sym, dq_sym)
    rhs = cs.vertcat(dq_sym, ddq_sym)
    f_dyn = cs.Function('f', [state_sym, ddq_sym], [state_sym + DT * rhs])
    
    kinDyn = KinDynComputations(robot.urdf, joints_name_list)
    H_b = cs.SX.eye(4)
    v_b = cs.SX.zeros(6)
    bias_forces = kinDyn.bias_force_fun()
    mass_matrix = kinDyn.mass_matrix_fun()
    h = bias_forces(H_b, q_sym, v_b, dq_sym)[6:]
    M = mass_matrix(H_b, q_sym)[6:,6:]
    tau_expr = M @ ddq_sym + h
    inv_dyn = cs.Function('inv_dyn', [state_sym, ddq_sym], [tau_expr])

    # Neural Network
    print("--- LOADING NEURAL NETWORK ---")
    net = NeuralNetwork(input_size=nx)
    # Load CasADi function
    brs_fun = net.create_casadi_function(
        robot_name=NN_MODEL_NAME, 
        NN_DIR=NN_DIR, 
        input_size=nx, 
        load_weights=True
    )
    
    # Build MPC
    print("--- BUILDING MPC (USE_NEURAL_NETWORK ={USE_NEURAL_NETWORK}) ---")
    opti, P_init, P_target, X, U = build_mpc_solver(
        N_MPC, nx, nu, nq, nv, f_dyn, inv_dyn, 
        tau_min_list, tau_max_list, lbx, ubx, q_lim, 
        brs_fun, q_des_val, USE_NEURAL_NETWORK
    )
    
    # Init Simulation
    r = RobotWrapper(robot.model, robot.collision_model, robot.visual_model)
    simu = RobotSimulator(conf_doublep, r)
    
    if SEED is not None:
        np.random.seed(SEED)
        print(f"Random seed set to: {SEED}")

    if USE_NEURAL_NETWORK:
        # here we generate randoma states, we check if the state is feasible 
        # for the neural network, otherwise we sample another random state until we find a feasible one
        print("--- SEARCHING FOR SAFE INITIAL STATE (BwRS Check) ---")
        max_retries = 10000
        found_safe = False

        for attempt in range(max_retries):
            
            # q1: Between -2.5 and 0.0 
            q1_rnd = np.random.uniform(-2.5, 0.0)
            q2_rnd = np.random.uniform(0.05, 1.0)
            # Velocity: Between -3 and 3
            dq1_rnd = np.random.uniform(-5.0, 5.0)
            dq2_rnd = np.random.uniform(-5.0, 5.0)
            
            candidate_state = np.array([q1_rnd, q2_rnd, dq1_rnd, dq2_rnd])

            # check if for the nn is safe
            safety_prob = brs_fun(candidate_state).full().item()
            
            # --- 3. Validation ---
            if safety_prob >= PROB_THRESHOLD:
                print(colored(f"Safe State Found after {attempt} attempts!", "green"))
                print(f"State: q=[{q1_rnd:.2f}, {q2_rnd:.2f}], v=[{dq1_rnd:.2f}, {dq2_rnd:.2f}]")
                print(f"Safety Probability: {safety_prob:.4f}")
                
                q0 = candidate_state[:nq]
                dq0 = candidate_state[nq:]
                x_curr = candidate_state
                found_safe = True
                break
                
        if not found_safe:
            print(colored("Unable to find a valid safe state. Using default state.", "red"))
            q0 = np.array([-1.5, 0.5]) # A definitely safe point
            dq0 = np.zeros(nv)
            x_curr = np.concatenate([q0, dq0])

    else:
        print("--- GENERATING RANDOM STATE (NO SAFETY CHECK) ---")
        q1_rnd = np.random.uniform(-2.5, 0.0)
        q2_rnd = np.random.uniform(0.05, 1.0)
        dq1_rnd = np.random.uniform(-5.0, 5.0) 
        dq2_rnd = np.random.uniform(-5.0, 5.0)
        
        q0 = np.array([q1_rnd, q2_rnd])
        dq0 = np.array([dq1_rnd, dq2_rnd])
        x_curr = np.concatenate([q0, dq0])
        
    print(colored(f"--- STARTING STATE CHECK ---", "cyan"))
    print(f"q0 = [{x_curr[0]:.4f}, {x_curr[1]:.4f}]")
    print(f"v0 = [{x_curr[2]:.4f}, {x_curr[3]:.4f}]")
    print("------------------------------")
        
    simu.init(q0, dq0)
    simu.display(q0)
    

    time.sleep(1) 
    
    SIM_STEPS = 500
    
    # --- LOGGING LISTS ---
    history_q = []
    history_v = []    # Added for plotting
    history_tau = []  # Added for plotting
    
    print("--- STARTING SIMULATION ---")
    
    # Warm Start Variables
    sol_X = None
    sol_U = None
    
    for t in range(SIM_STEPS):
        t_start = time.time()
        
        # A. Set Parameters
        opti.set_value(P_init, x_curr)
        opti.set_value(P_target, q_des_val)
        
        # B. WARM START STRATEGY
        if t == 0:
            scaling_factor = np.linspace(1.0, 0.0, N_MPC + 1)
            q_guess = np.tile(x_curr[:nq].reshape(-1, 1), (1, N_MPC + 1))
            # Velocity: scale to zero linearly
            dq_guess = x_curr[nq:].reshape(-1, 1) * scaling_factor.reshape(1, -1)
            
            x_guess = np.vstack((q_guess, dq_guess))
            
            opti.set_initial(X, x_guess)
            opti.set_initial(U, 0.0) 
            print("Init: Linear Decay Warm Start Applied")
            
        else:
            # Take old solution and shift back by 1
            if sol_X is not None:
                X_old = sol_X
                U_old = sol_U
                
                # Shift X: [x1, x2, ... xN, xN]
                X_guess = np.hstack((X_old[:, 1:], X_old[:, -1:]))
                # Shift U: [u1, u2, ... uN-1, 0] (or copy last)
                U_guess = np.hstack((U_old[:, 1:], np.zeros((nu, 1))))
                
                opti.set_initial(X, X_guess)
                opti.set_initial(U, U_guess)
        
        # SOLVE
        try:
            sol = opti.solve()
            
            # Extract pure numeric values
            sol_X = sol.value(X)
            sol_U = sol.value(U)
            
            # Optimal acceleration
            u_opt = sol_U[:, 0] # First action
            
            # Torque Calculation (Inv Dyn)
            tau_opt = inv_dyn(x_curr, u_opt).full().flatten()
            
        except Exception as e:
            print(colored(f"SOLVER FAILED at step {t}: {e}", "red"))
            tau_opt = np.zeros(nu) 
            break

        # D. SIMULATE (Pinocchio)
        # Integrate with small steps for realism
        dt_sim = 0.002
        n_substeps = int(DT / dt_sim)
        simu.simulate(tau_opt, dt_sim, n_substeps)
        
        # Update state
        x_curr = np.concatenate([simu.q, simu.v])
        
        # --- DATA LOGGING ---
        history_q.append(simu.q.copy())
        history_v.append(simu.v.copy())     # Store velocity
        history_tau.append(tau_opt.copy())  # Store torque
        
        # Timing for real-time visualization
        elapsed = time.time() - t_start
        if elapsed < DT:
            time.sleep(DT - elapsed)
            
        if t % 20 == 0:
            # Check neural constraint
            nn_val = brs_fun(sol_X[:, -1]).full().item()
            print(f"Step {t:03d} | Tau: {tau_opt.round(2)} | NN Val: {nn_val:.2f} | Dist: {np.linalg.norm(x_curr[:nq]-q_des_val):.2f}")

    print("Simulation Finished.")
    
    # --- SAFETY CHECK ---
    print("\n--- SAFETY CHECK ---")
    hist_q = np.array(history_q)
    min_q = np.min(hist_q, axis=0) # Finds the lowest value reached by q1 and q2

    # Check against limits
    is_safe = np.all(min_q >= q_lim)
    
    print(f"Q1 Min: {min_q[0]:.4f} (Limit: {q_lim[0]:.4f}) -> {'OK' if min_q[0] >= q_lim[0] else 'FAIL'}")
    print(f"Q2 Min: {min_q[1]:.4f} (Limit: {q_lim[1]:.4f}) -> {'OK' if min_q[1] >= q_lim[1] else 'FAIL'}")

    if is_safe:
        print(colored("SUCCESS: All wall constraints respected!", "green"))
    else:
        print(colored("FAILURE: Wall penetration detected!", "red"))

    # --- COMPLETE PLOTTING (SEQUENTIAL) ---
    hist_q = np.array(history_q)
    hist_v = np.array(history_v)
    hist_tau = np.array(history_tau)

    print("Opening POSITION plot... ")
    
    # 1. POSITION PLOT
    plt.figure(figsize=(10, 6))
    plt.plot(hist_q[:, 0], label='q1 (Shoulder)', linewidth=2)
    plt.plot(hist_q[:, 1], label='q2 (Elbow)', linewidth=2)
    # Plot Wall Limits explicitly
    plt.axhline(y=q_lim[0], color='r', linestyle='--', label=f'Wall Limit q1 ({q_lim[0]:.2f})')
    plt.axhline(y=q_lim[1], color='m', linestyle='--', label=f'Wall Limit q2 ({q_lim[1]:.2f})')
    plt.ylabel("Position [rad]")
    plt.xlabel("Simulation Steps")
    plt.title("1/3 - Joint Positions vs Wall Limits")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
     

    print("Opening VELOCITY plot... ")

    # 2. VELOCITY PLOT
    plt.figure(figsize=(10, 6))
    plt.plot(hist_v[:, 0], label='dq1', linewidth=1.5)
    plt.plot(hist_v[:, 1], label='dq2', linewidth=1.5)
    # Plot Max Velocity Limits
    plt.axhline(y=vMax[0], color='k', linestyle=':', label='Max Velocity')
    plt.axhline(y=-vMax[0], color='k', linestyle=':')
    plt.ylabel("Velocity [rad/s]")
    plt.xlabel("Simulation Steps")
    plt.title("2/3 - Joint Velocities")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
     

    print("Opening TORQUE plot...")

    # 3. TORQUE PLOT
    plt.figure(figsize=(10, 6))
    plt.plot(hist_tau[:, 0], label='tau1', linewidth=1.5)
    plt.plot(hist_tau[:, 1], label='tau2', linewidth=1.5)
    # Plot Max Torque Limits
    plt.axhline(y=tau_lim_abs, color='r', linestyle='--', label=f'Max Torque ({tau_lim_abs:.1f})')
    plt.axhline(y=-tau_lim_abs, color='r', linestyle='--')
    plt.ylabel("Torque [Nm]")
    plt.xlabel("Simulation Steps")
    plt.title("3/3 - Joint Torques")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    

if __name__ == "__main__":
    main()