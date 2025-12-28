#Data check for multiple datasets with different horizons 

import numpy as np
import matplotlib.pyplot as plt
import os

def compare_4_datasets(file_list):
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten() # Flatten to 1D array for easy indexing

    for i, filepath in enumerate(file_list):
        if not os.path.exists(filepath):
            axes[i].set_title(f"File not found:\n{os.path.basename(filepath)}", color='red')
            continue
            
        # Load data
        data = np.load(filepath)
        x = data['x']
        y = data['y']

        # Plot (assuming Single Pendulum: column 0 = q, column 1 = dq)
        axes[i].scatter(x[y==0, 0], x[y==0, 1], c='red', s=1, alpha=0.3, label='Unsafe')
        axes[i].scatter(x[y==1, 0], x[y==1, 1], c='green', s=3, alpha=0.6, label='Safe')
        
        axes[i].set_title(f"Horizon {os.path.basename(filepath)}")
        axes[i].set_xlabel('q [rad]')
        axes[i].set_ylabel('dq [rad/s]')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
       files_comparison = [
        'data/dataset_p_25.npz',
        'data/dataset_p_50.npz',
        'data/dataset_p_100.npz',
        'data/dataset_p_200.npz'
    ]
compare_4_datasets(files_comparison)