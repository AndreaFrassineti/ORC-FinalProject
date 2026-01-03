import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
import os

def check_q1_alignment_final(model_path, dataset_path):
    print(f"--- VERIFICA FINALE Q1 (Dinamica vs Geometria) ---")
    
    # 1. Carica Dati
    data = np.load(dataset_path)
    X_all = data['x'] 
    Y_all = data['y']
    
    # 2. Carica Modello
    model = NeuralNetwork(input_size=4, hidden_size=32, output_size=1)
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 3. FILTRO CRUCIALE PER ISOLARE Q1
    # Vogliamo vedere solo i crash causati dal braccio 1.
    # Quindi prendiamo solo i casi dove il braccio 2 (q2) è SICURO.
    # q2 > 0.2 (lontano dal muro -0.05) e dq2 basso.
    mask_isolate_q1 = (X_all[:, 1] > 0.2) & (np.abs(X_all[:, 3]) < 2.0)
    
    X_subset = X_all[mask_isolate_q1]
    Y_subset = Y_all[mask_isolate_q1]
    
    print(f"Punti totali: {len(X_all)}")
    print(f"Punti usati per Q1 (dove Q2 è safe): {len(X_subset)}")

    # 4. GRIGLIA Q1 (Allargata a -7 -> 7 per vedere tutto)
    q1_vals = np.linspace(-7, 7, 300)      # Posizione q1
    dq1_vals = np.linspace(-12, 12, 300)   # Velocità dq1
    Q1, DQ1 = np.meshgrid(q1_vals, dq1_vals)
    
    # Flatten per evitare rotazioni
    q1_flat = Q1.flatten()
    dq1_flat = DQ1.flatten()
    n_p = len(q1_flat)
    
    # Input Tensore: [q1_variabile, q2_fisso, dq1_variabile, dq2_fisso]
    # Fissiamo q2=0.5 (sicuro) e dq2=0.0
    q2_fixed = np.full(n_p, 0.5)
    dq2_fixed = np.zeros(n_p)
    
    inputs = np.stack([q1_flat, q2_fixed, dq1_flat, dq2_fixed], axis=1)
    inputs_t = torch.tensor(inputs, dtype=torch.float32)
    
    # 5. PREDIZIONE
    with torch.no_grad():
        probs = torch.sigmoid(model(inputs_t)).numpy()
    
    Z = probs.reshape(Q1.shape)

    # 6. PLOT
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Sfondo (Rete)
    im = ax.pcolormesh(Q1, DQ1, Z, cmap='RdYlGn', shading='auto', vmin=0, vmax=1)
    ax.contour(Q1, DQ1, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Punti Reali (Dataset)
    # Colonna 0 = q1, Colonna 2 = dq1
    safe_pts = X_subset[Y_subset == 1]
    crash_pts = X_subset[Y_subset == 0]
    
    ax.scatter(safe_pts[:, 0], safe_pts[:, 2], c='blue', s=15, alpha=0.5, label='Dataset Safe (1)')
    ax.scatter(crash_pts[:, 0], crash_pts[:, 2], c='red', marker='x', s=30, alpha=0.8, label='Dataset Crash (0)')

    # Muro Fisico Q1
    ax.axvline(x=-(np.pi+0.05), color='blue', linestyle='--', linewidth=3, label='Muro Fisico (-3.19)')

    ax.set_title(f"VERIFICA Q1: Dinamica vs Geometria\n(I punti rossi DEVONO cadere nelle zone rosse, anche se 'strane')")
    ax.set_xlabel("q1 [rad] (Posizione)")
    ax.set_ylabel("dq1 [rad/s] (Velocità)")
    ax.legend()
    plt.colorbar(im, label="Probabilità Feasible")
    plt.grid(True, alpha=0.3)
    
    plt.show()

if __name__ == "__main__":
    d_path = "data/dataset_dp_friend2_200.npz"
    m_path = "nn_models/model_dp_friend2_200.pt"
    
    check_q1_alignment_final(m_path, d_path)