import numpy as np
import matplotlib.pyplot as plt
from pendulum import Pendulum
from mpc import run_mpc_control 
import casadi as ca

def main():
    # 1. Setup of the simulation
    p = Pendulum()
    dt = 0.05
    sim_time = 10.0 # Seconds of simulation
    steps = int(sim_time / dt)
    
    # Choose the model N=100 that we saw being solid in training
    model_path = 'models/model_p_100.pt'
    
    # Initialize the CasADi optimizer
    opti, x_init_param, u_var = run_mpc_control(model_path, n_steps=20)
    
    # 2. Initial Conditions
    # Try a difficult position: almost horizontal (1.5 rad)
    x = np.array([1.5, 0.0]) 
    
    history_x = [x]
    history_u = []
    
    print(f"Starting simulation from x0 = {x}...")

    # 3. Main Loop
    for i in range(steps):
        # Set the current state parameter
        opti.set_value(x_init_param, x)
        
        try:
            # Solve the the optimization problem
            sol = opti.solve()
            u_control = sol.value(u_var)[0] # Choose the first action only (Receding Horizon)
        except:
            # If the optimizer fails, take the last valid solution or 0
            print(f"Warning: Optimizer failed at step {i}. Safety compromised!")
            u_control = 0.0
        
        # Apply the torque and update the physics
        x = p.dynamics_step(x, u_control, dt)
        
        history_x.append(x)
        history_u.append(u_control)

        if i % 20 == 0:
            print(f"Step {i}/{steps} - Pos: {x[0]:.2f}, Vel: {x[1]:.2f}")

    # 4. Visualization of Results
    plot_results(np.array(history_x), np.array(history_u), dt)

def plot_results(x, u, dt):
    t = np.linspace(0, len(u)*dt, len(u))
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    ax[0].plot(t, x[:-1, 0], label='Angolo q [rad]')
    ax[0].axhline(0, color='r', linestyle='--', alpha=0.3)
    ax[0].set_ylabel('q [rad]')
    ax[0].legend()
    
    ax[1].plot(t, x[:-1, 1], label='Velocità dq [rad/s]', color='orange')
    ax[1].set_ylabel('dq [rad/s]')
    ax[1].legend()
    
    ax[2].step(t, u, label='Coppia u [Nm]', color='green')
    ax[2].set_ylabel('u [Nm]')
    ax[2].set_xlabel('Tempo [s]')
    ax[2].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()