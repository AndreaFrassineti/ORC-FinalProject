import numpy as np
import matplotlib.pyplot as plt
import os

def check_double_pendulum_data(filepath='data/dataset_double_pendulum.npz'):
    if not os.path.exists(filepath):
        print(f"Errore: Il file {filepath} non esiste.")
        return

    data = np.load(filepath)
    x = data['x'] # [q1, q2, dq1, dq2]
    y = data['y']
    
    print(f"Totale campioni: {len(y)}")
    print(f"Campioni fattibili (Label 1): {np.sum(y)}")
    print(f"Campioni non fattibili (Label 0): {len(y) - np.sum(y)}")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Grafico 1: Primo giunto (q1 vs dq1)
    ax[0].scatter(x[y==0, 0], x[y==0, 2], c='red', s=2, alpha=0.3, label='Unsafe')
    ax[0].scatter(x[y==1, 0], x[y==1, 2], c='green', s=5, alpha=0.6, label='Safe')
    ax[0].set_title('Giunto 1: Posizione vs Velocità')
    ax[0].set_xlabel('q1 [rad]')
    ax[0].set_ylabel('dq1 [rad/s]')
    ax[0].legend()

    # Grafico 2: Secondo giunto (q2 vs dq2)
    ax[1].scatter(x[y==0, 1], x[y==0, 3], c='red', s=2, alpha=0.3, label='Unsafe')
    ax[1].scatter(x[y==1, 1], x[y==1, 3], c='green', s=5, alpha=0.6, label='Safe')
    ax[1].set_title('Giunto 2: Posizione vs Velocità')
    ax[1].set_xlabel('q2 [rad]')
    ax[1].set_ylabel('dq2 [rad/s]')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_double_pendulum_data()