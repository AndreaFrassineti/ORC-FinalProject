import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
import os

def plot_models(model_files):
    n_models = len(model_files)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10), squeeze=False)
    
    DELTA = 0.05
    WALL_Q1 = -(np.pi + DELTA) 
    WALL_Q2 = -(0.0 + DELTA)   

    # 1. Creiamo la griglia
    q_vals = np.linspace(-4, 4, 200)
    dq_vals = np.linspace(-10, 10, 200)
    Q, DQ = np.meshgrid(q_vals, dq_vals)
    
    
    q_flat = Q.flatten()
    dq_flat = DQ.flatten()
    n_points = len(q_flat)
    
    zeros = np.zeros(n_points)
    
    # When analyzing Q1, we fix Q2 at 0.5 (far from its wall at -0.05)
    q2_safe_val = np.full(n_points, 0.5) 
    
   # When analyzing Q2, we fix Q1 at 0.0 (far from its wall at -3.19)
    q1_safe_val = np.zeros(n_points) 

    
    input_np_q1 = np.stack([q_flat, q2_safe_val, dq_flat, zeros], axis=1)
    X_q1 = torch.tensor(input_np_q1, dtype=torch.float32)

    input_np_q2 = np.stack([q1_safe_val, q_flat, zeros, dq_flat], axis=1)
    X_q2 = torch.tensor(input_np_q2, dtype=torch.float32)

    # --- MODEL LOOP ---
    for col, model_path in enumerate(model_files):
        model_name = os.path.basename(model_path).replace("model_dp_", "").replace(".pt", "")
        
        # Access axes directly 
        ax1 = axes[0, col]
        ax2 = axes[1, col]

        if not os.path.exists(model_path):
            ax1.set_title(f"NOT FOUND\n{model_name}", color='red')
            print(f"File not found: {model_path}")
            continue

        # --- LOAD MODEL ---
        # Set hidden_size=32 to match saved weights
        model = NeuralNetwork(input_size=4, hidden_size=32, output_size=1)
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue
            
        model.eval()

        # --- PLOT ROW 1 (Q1 vs dQ1) ---
        with torch.no_grad():
            probs_flat = torch.sigmoid(model(X_q1)).numpy()
            Z1 = probs_flat.reshape(Q.shape)
        
        # Plot heatmap
        im1 = ax1.pcolormesh(Q, DQ, Z1, cmap='RdYlGn', shading='auto', vmin=0, vmax=1)
        # Decision boundary (50% probability)
        ax1.contour(Q, DQ, Z1, levels=[0.5], colors='black', linewidths=1.5)
        # Physical Wall line
        ax1.axvline(x=WALL_Q1, color='blue', linestyle='--', linewidth=2, label='Wall Limit')
        
        ax1.set_title(f"N={model_name}\nJoint 1 (Shoulder)\n(with q2=0.5 safe)")
        ax1.set_xlabel('q1 [rad]')
        if col == 0: ax1.set_ylabel('dq1 [rad/s]')

        # --- PLOT ROW 2 (Q2 vs dQ2) ---
        with torch.no_grad():
            probs_flat = torch.sigmoid(model(X_q2)).numpy()
            Z2 = probs_flat.reshape(Q.shape)
        
        im2 = ax2.pcolormesh(Q, DQ, Z2, cmap='RdYlGn', shading='auto', vmin=0, vmax=1)
        ax2.contour(Q, DQ, Z2, levels=[0.5], colors='black', linewidths=1.5)
        ax2.axvline(x=WALL_Q2, color='cyan', linestyle='--', linewidth=2, label='Wall Limit')
        
        ax2.set_title(f"Joint 2 (Elbow)\n(with q1=0.0 safe)")
        ax2.set_xlabel('q2 [rad]')
        if col == 0: ax2.set_ylabel('dq2 [rad/s]')

    plt.tight_layout()
    print("Displaying plots...")
    plt.show()

if __name__ == "__main__":
        models = [
        'models/model_dp_25.pt',
        'models/model_dp_50.pt',
        'models/model_dp_100.pt',
        'models/model_dp_200.pt'

]
    
print("Generating plots for Double Pendulum...")
plot_models(models)