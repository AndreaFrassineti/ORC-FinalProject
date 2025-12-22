import numpy as np
import matplotlib.pyplot as plt

def plot_dataset(filename):
    data = np.load(filename)
    x = data['x']
    y = data['y']
    
    plt.figure(figsize=(10, 6))
    # Unfeasible points (0) in red
    plt.scatter(x[y==0, 0], x[y==0, 1], c='red', s=5, label='Infeasible (0)', alpha=0.3)
    # Feasible points (1) in green
    plt.scatter(x[y==1, 0], x[y==1, 1], c='green', s=5, label='Feasible (1)', alpha=0.6)
    
    plt.xlabel('Joint Position (q) [rad]')
    plt.ylabel('Joint Velocity (dq) [rad/s]')
    plt.title(f'Viability Kernel - {filename}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Esempio d'uso
plot_dataset('data/dataset_pendulum.npz')