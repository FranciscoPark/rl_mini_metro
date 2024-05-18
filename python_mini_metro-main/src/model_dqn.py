import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the DQN network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the DQN
input_dim = 150  # Example input dimension (40 for passengers, 100 for connections, 10 for critical states)
output_dim = 3  # Number of actions: expand, delete, do nothing
dqn = DQN(input_dim, output_dim)

# Example state representation
passenger_counts = np.random.randint(0, 13, (10, 4)).flatten()  # Flattened 10x4 matrix
metro_config = np.random.randint(0, 2, (10, 10)).flatten()  # Flattened 10x10 adjacency matrix
critical_states = np.random.randint(0, 2, 10)  # Binary vector of critical states

# Combine into a single input vector
state = np.concatenate([passenger_counts, metro_config, critical_states])

# Convert to tensor and pass through the network
state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
q_values = dqn(state_tensor)

print("Q-values:", q_values)
