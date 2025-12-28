import numpy as np
import matplotlib.pyplot as plt
import os

def check_double_pendulum_data(filepath='data/dataset_dp_N.npz'):
    if not os.path.exists(filepath):
        print(f"Error: file {filepath} does not exist.")
        return

    data = np.load(filepath)
    x = data['x'] # [q1, q2, dq1, dq2]
    y = data['y']
    
    print(f"Total samples: {len(y)}")
    print(f"Feasible samples (Label 1): {np.sum(y)}")
    print(f"Unfeasible samples (Label 0): {len(y) - np.sum(y)}")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: First joint (q1 vs dq1)
    ax[0].scatter(x[y==0, 0], x[y==0, 2], c='red', s=2, alpha=0.3, label='Unsafe')
    ax[0].scatter(x[y==1, 0], x[y==1, 2], c='green', s=5, alpha=0.6, label='Safe')
    ax[0].set_title('Joint 1: Position vs Velocity')
    ax[0].set_xlabel('q1 [rad]')
    ax[0].set_ylabel('dq1 [rad/s]')
    ax[0].legend()

    # Plot 2: Second joint (q2 vs dq2)
    ax[1].scatter(x[y==0, 1], x[y==0, 3], c='red', s=2, alpha=0.3, label='Unsafe')
    ax[1].scatter(x[y==1, 1], x[y==1, 3], c='green', s=5, alpha=0.6, label='Safe')
    ax[1].set_title('Joint 2: Position vs Velocity')
    ax[1].set_xlabel('q2 [rad]')
    ax[1].set_ylabel('dq2 [rad/s]')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

# Plot the dataset
if __name__ == "__main__":
    check_double_pendulum_data('data/dataset_dp_10.npz')