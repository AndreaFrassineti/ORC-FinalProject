import torch
import torch.nn as nn
from casadi import MX, Function
import l4casadi as l4c
import os

class NeuralNetwork(nn.Module):
    """ 
    Replica ESATTA della rete del tuo amico.
    Struttura Semplice: Input -> [Linear 32] -> Tanh -> [Linear Output]
    Nessuna trasformazione trigonometrica (sin/cos).
    """
    def __init__(self, input_size, hidden_size=32, output_size=1, activation=nn.Tanh()):
        super(NeuralNetwork, self).__init__()
        
        # Architettura identica al codice "facile"
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size)
        )
        
        # Upper bound (moltiplicatore finale)
        self.ub = torch.ones((output_size, 1))
        
        # Inizializzazione pesi
        self.initialize_weights()

    def forward(self, x):
        # Sposta ub sullo stesso device di x
        self.ub = self.ub.to(x.device)
        
        # Gestione input shape (standard friend logic)
        if x.ndimension() == 1:
            x = x.view(1, -1)
        elif x.ndimension() == 2 and x.shape[0] == 4 and x.shape[1] != 4:
            x = x.T

        # Passaggio diretto senza feature engineering
        out = self.linear_stack(x) * self.ub
        return out

    def initialize_weights(self):
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias) 

    # --- Wrapper per L4CasADi (Necessario per l'OCP dopo) ---
    def create_casadi_function(self, robot_name, NN_DIR, input_size, load_weights=True):
        if load_weights:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            nn_name = os.path.join(NN_DIR, f'model_{robot_name}.pt')
            try:
                # Caricamento robusto (gestisce sia dict che state_dict diretto)
                checkpoint = torch.load(nn_name, map_location=device)
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    self.load_state_dict(checkpoint['model'])
                else:
                    self.load_state_dict(checkpoint)
                print(f"Weights loaded successfully from {nn_name}")
            except Exception as e:
                print(f"Warning: Could not load weights: {e}")

        state = MX.sym("x", input_size) 
        
        self.l4c_model = l4c.L4CasADi(self,
                                      device='cuda' if torch.cuda.is_available() else 'cpu',
                                      name=f'{robot_name}_model',
                                      build_dir=os.path.join(NN_DIR, f'nn_{robot_name}'))

        self.nn_model = self.l4c_model(state)
        self.nn_func = Function('nn_func', [state], [self.nn_model])
        return self.nn_func