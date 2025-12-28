import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
import os

def compare_4_nn_models(model_files, input_dim=2):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Grid for the test
    q_range = np.linspace(-np.pi, np.pi, 100)
    dq_range = np.linspace(-8, 8, 100)
    Q, DQ = np.meshgrid(q_range, dq_range)
    
    grid_points = []
    for i in range(len(q_range)):
        for j in range(len(dq_range)):
            if input_dim == 2:
                grid_points.append([Q[j, i], DQ[j, i]])
            else:
                grid_points.append([Q[j, i], 0.0, DQ[j, i], 0.0])
    
    X_test = torch.tensor(grid_points, dtype=torch.float32)

    for i, model_path in enumerate(model_files):
        if not os.path.exists(model_path):
            axes[i].set_title(f"Model not found:\n{model_path}", color='red')
            continue

        # Loading model
        model = NeuralNetwork(input_size=input_dim, hidden_size=32, output_size=1)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()

        # Inference
        with torch.no_grad():
            Z = model(X_test).numpy().reshape(100, 100)

        # Plot of the "safety heatmap"
        im = axes[i].pcolormesh(Q, DQ, Z, cmap='RdYlGn', shading='auto', alpha=0.8)
        # Black contour line (threshold 0.5)
        axes[i].contour(Q, DQ, Z, levels=[0.5], colors='black', linewidths=1.5)
        
        axes[i].set_title(f"NN Opinion - {os.path.basename(model_path)}")
        axes[i].set_xlabel('q [rad]')
        axes[i].set_ylabel('dq [rad/s]')
        fig.colorbar(im, ax=axes[i], label='Safety Score')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    modelli = [
        'models/model_p_25.pt',
        'models/model_p_50.pt',
        'models/model_p_100.pt',
        'models/model_p_200.pt'
    ]
    compare_4_nn_models(modelli)