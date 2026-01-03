import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
import os

def plot_both_joints_phase_portraits_final(model_files):
    n_models = len(model_files)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))
    
    if n_models == 1:
        axes = np.array([axes]).T

    # --- SETUP COMUNE ---
    DELTA = 0.05
    WALL_Q1 = -(np.pi + DELTA)
    WALL_Q2 = -(0.0 + DELTA)

    # 1. Creiamo la griglia
    q_vals = np.linspace(-4, 4, 200)
    dq_vals = np.linspace(-10, 10, 200)
    Q, DQ = np.meshgrid(q_vals, dq_vals)
    
    # 2. Appiattiamo
    q_flat = Q.flatten()
    dq_flat = DQ.flatten()
    n_points = len(q_flat)
    
    zeros = np.zeros(n_points)
    
    # --- FIX CRUCIALE: Valori "Safe" per le variabili fisse ---
    # Quando analizziamo Q1, fissiamo Q2 lontano dal suo muro (-0.05).
    # Mettiamo q2 = 0.5 (così siamo sicuri che il rosso dipenda solo da Q1)
    q2_safe_val = np.full(n_points, 0.5) 
    
    # Quando analizziamo Q2, fissiamo Q1 lontano dal suo muro (-3.19).
    # Mettiamo q1 = 0.0 (è al centro, sicurissimo)
    q1_safe_val = np.zeros(n_points) 

    # --- COSTRUZIONE INPUT ---
    
    # RIGA 1: Analisi Q1 (variamo col 0 e 2)
    # Input: [q1_variabile, q2_SAFE, dq1_variabile, dq2_fermo]
    input_np_q1 = np.stack([q_flat, q2_safe_val, dq_flat, zeros], axis=1)
    X_q1 = torch.tensor(input_np_q1, dtype=torch.float32)

    # RIGA 2: Analisi Q2 (variamo col 1 e 3)
    # Input: [q1_SAFE, q2_variabile, dq1_fermo, dq2_variabile]
    input_np_q2 = np.stack([q1_safe_val, q_flat, zeros, dq_flat], axis=1)
    X_q2 = torch.tensor(input_np_q2, dtype=torch.float32)

    # --- LOOP MODELLI ---
    for col, model_path in enumerate(model_files):
        model_name = os.path.basename(model_path).replace("model_dp_", "").replace(".pt", "")
        
        if not os.path.exists(model_path):
            axes[0, col].set_title(f"NOT FOUND\n{model_name}", color='red')
            continue

        model = NeuralNetwork(input_size=4, hidden_size=32, output_size=1)
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
        except: continue
        model.eval()

        # --- PLOT RIGA 1 (Q1) ---
        ax1 = axes[0, col]
        with torch.no_grad():
            probs_flat = torch.sigmoid(model(X_q1)).numpy()
            Z1 = probs_flat.reshape(Q.shape)
        
        ax1.pcolormesh(Q, DQ, Z1, cmap='RdYlGn', shading='auto', vmin=0, vmax=1)
        ax1.contour(Q, DQ, Z1, levels=[0.5], colors='black', linewidths=1.5)
        ax1.axvline(x=WALL_Q1, color='blue', linestyle='--', linewidth=2, label='Wall Limit')
        
        ax1.set_title(f"{model_name}\nJoint 1 (con q2=0.5 safe)")
        ax1.set_xlabel('q1 [rad]')
        if col == 0: ax1.set_ylabel('dq1 [rad/s]')

        # --- PLOT RIGA 2 (Q2) ---
        ax2 = axes[1, col]
        with torch.no_grad():
            probs_flat = torch.sigmoid(model(X_q2)).numpy()
            Z2 = probs_flat.reshape(Q.shape)
        
        ax2.pcolormesh(Q, DQ, Z2, cmap='RdYlGn', shading='auto', vmin=0, vmax=1)
        ax2.contour(Q, DQ, Z2, levels=[0.5], colors='black', linewidths=1.5)
        ax2.axvline(x=WALL_Q2, color='cyan', linestyle='--', linewidth=2, label='Wall Limit')
        
        ax2.set_title(f"Joint 2 (con q1=0.0 safe)")
        ax2.set_xlabel('q2 [rad]')
        if col == 0: ax2.set_ylabel('dq2 [rad/s]')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    models = [
        'nn_models/model_dp_friend2_200.pt', 
        'nn_models/model_dp_friend_200.pt', 
        'nn_models/model_dp_friend_50.pt'
    ]
    plot_both_joints_phase_portraits_final(models)