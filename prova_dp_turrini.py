import casadi as cs
import numpy as np
from example_robot_data import load
from adam.casadi.computations import KinDynComputations
import os

def generate_dataset_aligned(n_samples=5000, N=50):
    # 0. Create data folder if it doesn't exist
    folder = "data"
    if not os.path.exists(folder):
        os.makedirs(folder)

    
    SOLVER_TOLERANCE = 1e-4
    dt = 0.010 
    VEL_BOUNDS_SCALING_FACTOR = 1.0
    TORQUE_BOUNDS_SCALING_FACTOR = 1.0 # Questo limita molto la coppia rispetto alla gravità reale
    
    qMin = np.array([-2.0*np.pi, -2.0*np.pi])
    qMax = -qMin
    vMax = np.array([10.0, 10.0]) * VEL_BOUNDS_SCALING_FACTOR
    vMin = -vMax
    
    tauMin = np.array([-1.0, -1.0]) * TORQUE_BOUNDS_SCALING_FACTOR
    tauMax = -tauMin

    # Definition of a physical constraint (il blocco meccanico/muro)
    DELTA = 0.05
    q_lim = np.array([-(np.pi+DELTA), -(0.0+DELTA)])

    print("--- LOADING ROBOT MODEL ---")
    robot = load("double_pendulum")
    joints_name_list = [s for s in robot.model.names[1:]] 
    
    nq = robot.model.nq
    nx = 2 * nq
    
    # KinDyn Setup
    kinDyn = KinDynComputations(robot.urdf, joints_name_list)
    
    # --- COSTRUZIONE FUNZIONI CASADI (Fisse per tutti i cicli) ---
    q   = cs.SX.sym('q', nq)
    dq  = cs.SX.sym('dq', nq)
    ddq = cs.SX.sym('ddq', nq)
    state = cs.vertcat(q, dq)
    rhs   = cs.vertcat(dq, ddq)
    f = cs.Function('f', [state, ddq], [rhs])

    # Inverse Dynamics Function
    H_b = cs.SX.eye(4)
    v_b = cs.SX.zeros(6)
    bias_forces = kinDyn.bias_force_fun()
    mass_matrix = kinDyn.mass_matrix_fun()
    
    h_term = bias_forces(H_b, q, v_b, dq)[6:]
    M_term = mass_matrix(H_b, q)[6:, 6:]
    tau_expr = M_term @ ddq + h_term
    inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau_expr])

    # Pre-compute bounds lists
    lbx = qMin.tolist() + vMin.tolist()
    ubx = qMax.tolist() + vMax.tolist()
    tau_min_list = tauMin.tolist()
    tau_max_list = tauMax.tolist()

    # --- FUNZIONE SOLVER ---
    def solve_ocp_friend_style(x_init):
        opti = cs.Opti()
        
        # Parameter for initial state (come nel codice del tuo amico)
        param_x_init = opti.parameter(nx)
        
        # Variabili decisionale come LISTE (Struttura del tuo amico)
        X, U = [], []
        
        # Stato iniziale (variabile 0)
        X += [opti.variable(nx)] 
        # Non applica bounds pos/vel sullo stato iniziale, solo uguaglianza dopo
        
        # Loop stati successivi
        for k in range(1, N+1):
            X += [opti.variable(nx)]
            opti.subject_to(opti.bounded(lbx, X[-1], ubx))
            
        # Loop controlli e vincoli dinamici
        for k in range(N):
            U += [opti.variable(nq)]
            
            # 1. Dynamics constraints (Eulero esplicito diretto)
            opti.subject_to(X[k+1] == X[k] + dt * f(X[k], U[k]))
            
            # 2. Torque constraints
            opti.subject_to(opti.bounded(tau_min_list, inv_dyn(X[k], U[k]), tau_max_list))
            
            # 3. Physical constraints (q > q_lim)
            opti.subject_to(X[k][:nq] >= q_lim)

        # Initial condition constraint
        opti.subject_to(X[0] == param_x_init)
        
        # Terminal constraint: stationary point with zero velocity
        opti.subject_to(X[-1][nq:] == 0.0)
        
        opti.minimize(1) # Feasibility problem

        # Solver options (Copiati dal codice del tuo amico)
        opts = {
            "error_on_fail": False,
            "ipopt.print_level": 0,
            "ipopt.tol": SOLVER_TOLERANCE,
            "ipopt.constr_viol_tol": SOLVER_TOLERANCE,
            "ipopt.compl_inf_tol": SOLVER_TOLERANCE,
            "print_time": 0,
            "detect_simple_bounds": True,
            "ipopt.max_iter": 1000
        }
        opti.solver("ipopt", opts)

        try:
            opti.set_value(param_x_init, x_init)
            opti.solve()
            return 1
        except:
            return 0

    # --- DATASET GENERATION LOOP ---
    print(f"Generazione Dataset ({n_samples} campioni) stile 'Friend Logic'...")
    data_x, data_y = [], []

    import time
    start_time = time.time()

    for i in range(n_samples):
        # Sampling q1, q2 in [-pi, pi] and dq1, dq2 in [-vmax, vmax]
        # Nota: Uso i limiti definiti sopra per coerenza
        q_rand = np.random.uniform(-qMax, qMax) # Attenzione: qMax qui è positivo
        dq_rand = np.random.uniform(-vMax, vMax)
        
        x_start = np.concatenate([q_rand, dq_rand])
        
        label = solve_ocp_friend_style(x_start)
        
        data_x.append(x_start)
        data_y.append(label)
        
        if (i+1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {i+1}/{n_samples} | Label 1 Found: {sum(data_y)} | Time: {elapsed:.1f}s")

    # Save the dataset
    filename = f'dataset_dp_friend2_{N}.npz'
    filepath = os.path.join(folder, filename)
    np.savez(filepath, x=np.array(data_x), y=np.array(data_y))
    print(f"Dataset saved in: {filepath}")

if __name__ == "__main__":
    generate_dataset_aligned(n_samples=5000, N=200)