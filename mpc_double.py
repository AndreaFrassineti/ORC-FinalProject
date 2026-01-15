import casadi as cs
import numpy as np
import torch
import pinocchio as pin
from example_robot_data import load
from adam.casadi.computations import KinDynComputations
import l4casadi as l4c
import time
import os
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork 
from orc.utils.robot_wrapper import RobotWrapper
from orc.utils.robot_simulator import RobotSimulator
from orc.utils.viz_utils import addViewerSphere, applyViewerConfiguration

class SimConfig:
    def __init__(self, nq):
        self.nq = nq
        self.q0 = np.zeros(nq)
        self.dt = 0.01 # Passo di simulazione (lo allineiamo al DT dell'MPC)
        self.simulate_coulomb_friction = 0
        self.simulation_type = 'euler'
        self.tau_coulomb_max = np.zeros(nq)
        self.randomize_robot_model = 0
        self.use_viewer = True
        self.which_viewer = 'meshcat'
        self.simulate_real_time = 1
        self.show_floor = False
        self.DISPLAY_T = 0.02

# --- CONFIGURATION ---
ROBOT_NAME = "double_pendulum"
# Note: The logic in neural_network.py expects the file to be named "model_{robot_name}.pt" inside NN_DIR
# So we define the suffix here to match your file "model_dp_50.pt"
MODEL_SUFFIX = "dp_50" 
NN_DIR = "models"      # Folder where the .pt file is located
BUILD_DIR = "nn_build" # Temporary folder for L4CasADi compilation

N_MPC = 30     # MPC Horizon (can be shorter than training horizon!)
DT = 0.010     # Same DT as training
SIM_STEPS = 300 # Simulation duration

# Safety threshold.
# The network outputs logits. Sigmoid(0) = 0.5.
# > 0.0 implies probability > 50%.
# > 2.0 implies probability > 88% (more conservative).
SAFETY_THRESHOLD = 0.5 

print("--- LOADING ROBOT MODEL ---")
robot = load(ROBOT_NAME)
model = robot.model
data = model.createData()
joints_name_list = [s for s in robot.model.names[1:]] # skip the first name because it is "universe"

nq = model.nq
nv = model.nv
nx = 2 * nq
nu = nv

conf = SimConfig(nq)

# --- LOAD THE BACKWARD REACHABLE SET (NEURAL NETWORK) ---
# Here we modify the "Train" call to use the NeuralNetwork class correctly
# as defined in your neural_network.py file.
print("--- LOADING NEURAL NETWORK & L4CasADi ---")
net = NeuralNetwork(input_size=nx)

# This function (from your neural_network.py) loads weights and creates the CasADi function
# It looks for: os.path.join(NN_DIR, f'model_{robot_name}.pt')
# So we pass "dp_50" as robot_name so it finds "models/model_dp_50.pt"
backward_set = net.create_casadi_function(robot_name=MODEL_SUFFIX, 
                                          NN_DIR=NN_DIR, 
                                          input_size=nx, 
                                          load_weights=True)


def build_mpc_with_nn(N, nx, nu, nq, nv, f, inv_dyn, tau_min, tau_max, lbx, ubx, q_lim, backward_set_func, target_q):
   
    opti = cs.Opti()
    
    # optimization variables
    X = opti.variable(nx, N+1)
    U = opti.variable(nu, N)
    P_init = opti.parameter(nx)
    
    # cost
    cost = 0
    
    # weights of the cost function
    W_pos = 100.0   # position
    W_vel = 1e-8    # velocity
    W_acc = 1e-4   # acceleration
    
    for i in range(N):
        # dynamics
        opti.subject_to(X[:, i+1] == f(X[:, i], U[:, i]))
        
        # add torque constraints 
        # Using the KinDyn inverse dynamics passed as argument
        opti.subject_to(opti.bounded(tau_min, inv_dyn(X[:, i], U[:, i]), tau_max))
        
        # Physical constraints (q > q_lim)
        opti.subject_to(X[:nq, i+1] >= q_lim)
        
    
        # error on the position
        e_pos = X[:nq, i] - target_q
        v_k = X[nq:, i]
        a_k = U[:, i]

        cost += W_pos * cs.sumsqr(e_pos) 
        cost += W_vel * cs.sumsqr(v_k) 
        cost += W_acc * cs.sumsqr(a_k)

    # Initial constraint: X0 must be equal to the parameter P_init
    opti.subject_to(X[:, 0] == P_init)
    
    # Bounds on joint position and velocity
    # we don't limit X[0], because it's the initial sampled state 
    # X[:, 1:] take each row (pos+vel) and every column except of column 0 (state at 0)
    opti.subject_to(opti.bounded(lbx, X[:, 1:], ubx))

    
    # Terminal constraint (set S): both joint velocities are zero at final time
    # opti.subject_to(X[nv:, N] == 0) 
    # in my opinion this shouldn't be here, because I built the other one on purpose
    # (Leaving it commented out as per your intuition)

    # Terminal constraint (Neural Network)
    # We apply the learned function to the final state X[:, N]
    opti.subject_to(backward_set_func(X[:, N]) >= SAFETY_THRESHOLD)
    
    opti.minimize(cost)
    
    # Solver options (IPOPT for non-linearity)
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-3, 'detect_simple_bounds': True}
    opti.solver('ipopt', opts)
    
    return opti, P_init, X, U

def main_mpc():
    
    # Limits (copied from your generation script)
    # The wall
    q_lim_phys = np.array([-(np.pi+0.05), -(0.0+0.05)]) 
    
    # Target: We want to go VERY close to the wall.
    # E.g. 0.1 rad distance from the wall
    q_target = q_lim_phys + np.array([0.1, 0.1]) 
    
    print(f"Target Position: {q_target}")

    # Torque Limits
    # Assuming the same values used in data generation
    # You might want to retrieve exact values, but here is a reasonable override or calculation
    tau_lim_val = 6.0 
    tau_min = [-tau_lim_val] * nu
    tau_max = [tau_lim_val] * nu

    # State bounds (infinity for now, except physical ones handled in constraints)
    lbx = [-np.inf]*nx
    ubx = [np.inf]*nx
    
    # --- CasADi Dynamics Setup (Preserving KinDyn) ---
    q = cs.SX.sym('q', nq)
    dq = cs.SX.sym('dq', nv)
    ddq = cs.SX.sym('ddq', nv)
    state = cs.vertcat(q, dq)
    rhs = cs.vertcat(dq, ddq)
    f = cs.Function('f', [state, ddq], [state + DT * rhs])

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
    
    
    # --- BUILD MPC SOLVER ---
    print("--- Building MPC Solver ---")
    opti, P_init, X_var, U_var = build_mpc_with_nn(
        N_MPC, nx, nu, nq, nv, f, inv_dyn, 
        tau_min, tau_max, lbx, ubx, q_lim_phys, 
        backward_set, q_target
    )

    # --- SIMULATION LOOP ---
    # Initial state: Pendulum at rest far from the wall
    current_x = np.array([0.0, 0.0, 0.0, 0.0]) 
    
    history_x = [current_x]

    r = RobotWrapper(robot.model, robot.collision_model, robot.visual_model)
    simu = RobotSimulator(conf, r)
    simu.init(current_x[:nq], current_x[nq:])
    simu.display(current_x[:nq])

    # Creazione sfera target verde
    pin.forwardKinematics(model, data, q_target)
    pin.updateFramePlacements(model, data)
    ee_id = model.nframes - 1
    ee_pos = data.oMf[ee_id].translation
    
    addViewerSphere(r.viz, 'world/target', 0.1, [0, 1, 0, 0.5])
    applyViewerConfiguration(r.viz, 'world/target', ee_pos.tolist() + [0, 0, 0, 1])
    
    print("Opening Meshcat...")
    time.sleep(1.0) 
    
    print("\n--- Starting MPC Simulation ---")
    for t in range(SIM_STEPS):
        t0 =time.time()
        opti.set_value(P_init, current_x)
        
        # Warm start (optional but recommended)
        # opti.set_initial(X_var, ...)
        
        try:
            sol = opti.solve()
            
            # 1. Prendi l'accelerazione ottimale (ddq) dal solver
            u_acc_opt = sol.value(U_var)[:, 0]
            
            # 2. Calcola la coppia (tau) necessaria per quella accelerazione
            # Usiamo la funzione inv_dyn che hai già creato fuori dal loop
            tau_opt = inv_dyn(current_x, u_acc_opt).full().flatten()
            
            # 3. SIMULAZIONE FISICA REALE (Come il tuo amico)
            # Invece di fare next_x = f(...), usiamo il simulatore di Pinocchio
            dt_sim_physics = 0.002  # Passo fine per la fisica
            n_substeps = int(DT / dt_sim_physics) # Quanti passi fare (0.01 / 0.002 = 5 passi)
            
            # Applichiamo la coppia al robot
            simu.simulate(tau_opt, dt_sim_physics, n_substeps)
            
            # 4. Leggiamo dove è finito davvero il robot
            q_real = simu.q
            v_real = simu.v
            current_x = np.concatenate([q_real, v_real])
            
            history_x.append(current_x)

            # Display e Timing
            simu.display(current_x[:nq])
            
            t1 = time.time()
            if (t1 - t0) < DT:
                time.sleep(DT - (t1 - t0))
            
            if t % 20 == 0:
                print(f"Step {t}: q={current_x[:2].round(2)} | Tau={tau_opt.round(2)}")
            
                
        except Exception as e:
            print(f"Solver Failed at step {t}!")
            # Fallback or break
            break

    # --- PLOTTING ---
    hist_np = np.array(history_x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(hist_np[:, 0], label="q1")
    plt.plot(hist_np[:, 1], label="q2")
    # Draw the wall and the target
    plt.axhline(y=q_lim_phys[0], color='r', linestyle='--', label="Wall q1")
    plt.axhline(y=q_target[0], color='g', linestyle=':', label="Target q1")
    plt.legend()
    plt.title("MPC with Learned Backward Reachable Set")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main_mpc()