import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
import os

def compare_4_nn_models(model_files, input_dim=2, title_prefix=""):

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Generate a grid of points for evaluation
    q_range = np.linspace(-np.pi, np.pi, 100)
    dq_range = np.linspace(-8, 8, 100)
    Q, DQ = np.meshgrid(q_range, dq_range)
    
    grid_points = []
    for i in range(len(q_range)):
        for j in range(len(dq_range)):
            if input_dim == 2:
                # [q, dq]
                grid_points.append([Q[j, i], DQ[j, i]])
            else:
                # [q1, q2=0, dq1, dq2=0] - A slice for the double pendulum
                grid_points.append([Q[j, i], 0.0, DQ[j, i], 0.0])
    
    X_test = torch.tensor(grid_points, dtype=torch.float32)

    for i, model_path in enumerate(model_files):
        if i >= 4: break # 4 subplots
        
        if not os.path.exists(model_path):
            axes[i].set_title(f"Model not found:\n{model_path}", color='red')
            continue

        # Load the model
        model = NeuralNetwork(input_size=input_dim, hidden_size=32, output_size=1)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()

        with torch.no_grad():
            Z = model(X_test).numpy().reshape(100, 100)

        im = axes[i].pcolormesh(Q, DQ, Z, cmap='RdYlGn', shading='auto', alpha=0.8)
        axes[i].contour(Q, DQ, Z, levels=[0.5], colors='black', linewidths=1.5)
        
        axes[i].set_title(f"{title_prefix}\n{os.path.basename(model_path)}")
        axes[i].set_xlabel('q1 [rad]')
        axes[i].set_ylabel('dq1 [rad/s]')
        fig.colorbar(im, ax=axes[i])

    plt.tight_layout()

if __name__ == "__main__":
    # 1. Singol pendulum models 
    single_pendulum_models = [
        'models/model_p_25.pt',
        'models/model_p_50.pt',
        'models/model_p_100.pt',
        'models/model_p_200.pt'
    ]

    # 2. Double pendulum models
    double_pendulum_models = [
        'models/model_dp_25.pt',
        'models/model_dp_50.pt',
        'models/model_dp_100.pt',
        'models/model_dp_200.pt'
    ]

    print("Generation for the Single Pendulum...")
    compare_4_nn_models(single_pendulum_models, input_dim=2, title_prefix="Single Pendulum")
    
    print("Generation for the Double Pendulum...")
    compare_4_nn_models(double_pendulum_models, input_dim=4, title_prefix="Double Pendulum (q2=0)")

    plt.show()