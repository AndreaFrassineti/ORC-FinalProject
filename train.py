import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from neural_network import NeuralNetwork


# --- 2. FUNZIONE DI TRAINING ---
def train_friend_style(dataset_path, model_save_path, epochs=100):
    # Controllo esistenza file
    if not os.path.exists(dataset_path):
        print(f"Dataset NON trovato: {dataset_path} -> Salto questo task.")
        return

    print(f"\n>>> Caricamento dataset: {dataset_path}")
    data = np.load(dataset_path)
    X_np = data['x']
    y_np = data['y']

    # Converti in Tensor
    states = torch.tensor(X_np, dtype=torch.float32)
    labels = torch.tensor(y_np, dtype=torch.float32).reshape(-1, 1)
    
    # Statistiche rapide
    print(f"    Campioni totali: {len(states)}")
    n_ones = torch.sum(labels == 1).item()
    print(f"    Punti 'Feasible' (1): {n_ones} ({n_ones/len(states):.1%})")

    # Dataset e Split (80% Train, 20% Test)
    full_dataset = TensorDataset(states, labels)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # DataLoader
    batch_size = 32 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Setup Modello
    input_size = 4  
    hidden_size = 32
    output_size = 1
    
    model = NeuralNetwork(input_size, hidden_size, output_size, activation=nn.Tanh())
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f" Avvio training per {epochs} epoche...")

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validazione ogni 20 epoche
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for X_test, y_test in test_loader:
                    X_test, y_test = X_test.to(device), y_test.to(device)
                    probs = torch.sigmoid(model(X_test))
                    preds = (probs > 0.5).float()
                    total += y_test.size(0)
                    correct += (preds == y_test).sum().item()
            
            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total
            print(f"    Epoch [{epoch + 1}/{epochs}] | Loss: {avg_loss:.4f} | Test Acc: {accuracy:.2%}")

    # Salvataggio
    save_dir = os.path.dirname(model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save({'model': model.state_dict()}, model_save_path)
    print(f"     Modello salvato in: {model_save_path}")

# --- 3. MAIN CONFIGURATION ---
def train_everything():
    # LISTA DEI TASK AGGIORNATA
    tasks = [
        {"dataset": "data/dataset_dp_25.npz",  "model": "models/model_dp_25.pt",  "epochs": 200},
        #{"dataset": "data/dataset_dp_50.npz",  "model": "models/model_dp_50.pt",  "epochs": 200},
        #{"dataset": "data/dataset_dp_100.npz", "model": "models/model_dp_100.pt", "epochs": 200},
        #{"dataset": "data/dataset_dp_200.npz", "model": "models/model_dp_200.pt", "epochs": 200},
        
    ]

    print("--- INIZIO TRAINING MULTIPLO ---")
    for i, task in enumerate(tasks):
        print(f"\n🔹 TASK {i+1}/{len(tasks)}")
        train_friend_style(task["dataset"], task["model"], task["epochs"])
    print("\n--- TUTTI I TRAINING COMPLETATI ---")

if __name__ == "__main__":
    train_everything()