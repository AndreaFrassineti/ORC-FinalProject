import os
import torch
import numpy as np
import casadi as cs
from casadi import DM
from neural_network import NeuralNetwork
from data_generation_dp import get_solver_instance

def compare_real_and_nn(robot_name, nn_model_dir, input_size, x0, n_tests, horizon):
    
    # Load the Neural network
    print(f"--- Loading CasADi model for {robot_name} ---")
    net = NeuralNetwork(input_size=input_size)
    nn_func = net.create_casadi_function(robot_name, nn_model_dir, input_size)

    # Solver by the file data_generation_dp.py where dataset are generated
    print("\n--- Setup OCP Solver (Imported from data_generation) ---")
    opti, param_x_init, X_var, U_var, nq, vMax = get_solver_instance(N=horizon)
    
    # comparison loop
    print(f"\nStarting comparison on {n_tests} test...")
    correct_matches = 0
    
    # Use x0 only as the first point, then randomize
    current_x = x0

    for i in range(n_tests):
        opti.set_value(param_x_init, current_x)

        # apply the same warm start of data generation
        scaling_factor = np.linspace(1.0, 0.0, horizon + 1)
        q_guess = np.tile(current_x[:nq].reshape(-1, 1), (1, horizon + 1))
        dq_guess = current_x[nq:].reshape(-1, 1) * scaling_factor.reshape(1, -1)
        x_guess = np.vstack((q_guess, dq_guess))
        
        opti.set_initial(X_var, x_guess)
        opti.set_initial(U_var, 0.0)

        real_label = 0
        try:
            opti.solve()
            real_label = 1
        except:
            real_label = 0

        # same state using neural network
        nn_output_val = float(nn_func(DM(current_x)))
        nn_label = 1 if nn_output_val > 0.5 else 0

        # --- C. Stampa ---
        print(f"Test {i + 1}/{n_tests} | State: {np.round(current_x, 2)}")
        print(f"  Real (OCP): {real_label}")
        print(f"  Neural Network: {nn_output_val:.4f}")

        if real_label != nn_label:
            print("  >>> MISMATCH <<<")
        else:
            print("  [OK] Match")
            correct_matches += 1
        print("-" * 30)

        # Generate new states using the same ranges as your data generator
        q1_rand = np.random.uniform(-3.5, -1.5)
        q2_rand = np.random.uniform(-0.5, 1.5)
        # here we use lower bound of velocity to generate
        dq_rand = np.random.uniform(-6.0, 6.0, size=2) 
        
        current_x = np.concatenate([np.array([q1_rand, q2_rand]), dq_rand])

    accuracy = (correct_matches / n_tests) * 100
    print(f"\nFinal result: accuracy {accuracy:.1f}% ({correct_matches}/{n_tests})")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    nn_model_dir = os.path.join(current_dir, "models")
    robot_name = "dp_25"
    input_size = 4
    x0 = np.array([0.0, 0.0, 0.0, 0.0]) 
    
    NUM_TESTS = 200
    TRAINING_HORIZON = 25

    compare_real_and_nn(robot_name, nn_model_dir, input_size, x0, NUM_TESTS, TRAINING_HORIZON)