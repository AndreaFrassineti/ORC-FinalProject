import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
import os

def debug_final_alignment(model_path, dataset_path):
    # 1. CARICAMENTO DATI
    print(f"Loading data from {dataset_path}...")
    data = np.load(dataset_path)
    X_all = data['x'] 
    Y_all = data['y']
    
    # 2. CARICAMENTO MODELLO
    model = NeuralNetwork(input_size=4, hidden_size=32, output_size=1)
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 3. FILTRO DATI (Per isolare Q1)
    # Prendiamo punti dove Q2 è sicuro (> 0.2) e fermo (dq2 ~ 0)
    # Così se c'è un crash, è colpa di Q1.
    mask_isolate_q1 = (X_all[:, 1] > 0.2) & (np.abs(X_all[:, 3]) < 2.0)
    
    X_subset = X_all[mask_isolate_q1]
    Y_subset = Y_all[mask_isolate_q1]
    
    print(f"Punti totali dataset: {len(X_all)}")
    print(f"Punti usati per il plot (Q2 safe): {len(X_subset)}")

    # 4. GENERAZIONE GRIGLIA (Metodo "Flatten" - A prova di rotazione)
    # Allarghiamo il range come richiesto (-7 a 7)
    q_vals = np.linspace(-7, 7, 300)      # Asse X
    dq_vals = np.linspace(-12, 12, 300)   # Asse Y
    Q, DQ = np.meshgrid(q_vals, dq_vals)
    
    # Appiattiamo le matrici in vettori 1D
    q_flat = Q.flatten()
    dq_flat = DQ.flatten()
    
    # Creiamo l'input batch: [q_flat, q2_fixed, dq_flat, dq2_fixed]
    # q2_fixed = 0.5 (sicuro), dq2_fixed = 0.0
    n_points = len(q_flat)
    zeros = np.zeros(n_points)
    safe_q2 = np.full(n_points, 0.5) 
    
    # Costruiamo il tensor (N, 4)
    # Input order: q1, q2, dq1, dq2
    input_tensor = np.stack([q_flat, safe_q2, dq_flat, zeros], axis=1)
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
    
    # 5. PREDIZIONE
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).numpy()
    
    # Reshape usando la forma originale della meshgrid Q
    # Questo è il passaggio chiave che prima era sbagliato
    Z = probs.reshape(Q.shape)

    # 6. PLOT
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sfondo (Rete)
    # Usa pcolormesh con le stesse coordinate Q, DQ usate per generare l'input
    cm = ax.pcolormesh(Q, DQ, Z, cmap='RdYlGn', shading='auto', alpha=0.6, vmin=0, vmax=1)
    ax.contour(Q, DQ, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Punti (Dataset)
    # Safe = Blu
    safe_pts = X_subset[Y_subset == 1]
    ax.scatter(safe_pts[:, 0], safe_pts[:, 2], c='blue', s=15, alpha=0.5, label='Dataset Safe')
    
    # Crash = Rossi
    crash_pts = X_subset[Y_subset == 0]
    ax.scatter(crash_pts[:, 0], crash_pts[:, 2], c='red', marker='x', s=30, label='Dataset Crash')

    # Linee Muri
    ax.axvline(x=-(np.pi+0.05), color='blue', linestyle='--', linewidth=2, label='Muro Q1')
    
    ax.set_xlabel("q1 (Posizione)")
    ax.set_ylabel("dq1 (Velocità)")
    ax.set_title("Verifica Allineamento: Rete (Sfondo) vs Dati Reali (Punti)\n(Se vedi rosso su rosso e blu su verde, funziona)")
    ax.legend()
    plt.colorbar(cm, label="Probabilità Feasibility")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Usa i percorsi corretti
    d_path = "data/dataset_dp_friend2_200.npz"
    m_path = "nn_models/model_dp_friend2_200.pt"
    
    debug_final_alignment(m_path, d_path)