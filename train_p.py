import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from neural_network import NeuralNetwork # Import the NeuralNetwork class

def train_viability_model(dataset_path, model_save_name, epochs=50, batch_size=100):
    # 1. Load the data .npz
    if not os.path.exists(dataset_path):
        print(f"Error: The file {dataset_path} does not exist!")
        return

    data = np.load(dataset_path)
    X_raw = data['x']
    y_raw = data['y']

    # Convert to PyTorch tensors
    X = torch.tensor(X_raw, dtype=torch.float32)
    y = torch.tensor(y_raw, dtype=torch.float32).reshape(-1, 1)

    print(f"Dataset loaded: {X.shape[0]} samples.")
    # 2. Initialize the Network
    # Input_size: 2 for single pendulum, 4 for double pendulum
    input_dim = X.shape[1] 
    model = NeuralNetwork(input_size=input_dim, hidden_size=32, output_size=1)
    
    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Use BCEWithLogitsLoss if the last layer doesn't have Sigmoid,
    # but your class has Tanh. For binary classification with Tanh,
    # we can scale or use MSE. Using MSE for simplicity on label regression.
    criterion = nn.MSELoss() 

    # 3. Training Loop
    print(f"Starting training for {model_save_name}...")
    model.train()

    for epoch in range(epochs):
        # Shuffle the data at each epoch
        indices = torch.randperm(X.size(0))
        X, y = X[indices], y[indices]

        epoch_loss = 0
        for i in range(0, X.size(0), batch_size):
            batch_x = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/(X.size(0)/batch_size):.6f}")

    # 4. Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    save_path = os.path.join('models', model_save_name)
    # The format required by your 'load_weights' function in neural_network.py
    torch.save({'model': model.state_dict()}, save_path)
    print(f"Training completed! Model saved at: {save_path}")

if __name__ == "__main__":
    # FOR SINGLE PENDULUM (N=100)
    train_viability_model(
        dataset_path='data/dataset_p_100.npz', 
        model_save_name='model_p_100.pt'
    )
    
    # FOR DOUBLE PENDULUM
    # train_viability_model(
    #     dataset_path='data/dataset_dp_50.npz', 
    #     model_save_name='model_dp.pt'
    # )