import random
import torch
import numpy as np

# Define the possible actions
ACTIONS = ['expand_line', 'delete_line', 'do_nothing']

# Function to select an action using epsilon-greedy policy
def select_action(state, epsilon, dqn):
    if random.uniform(0, 1) < epsilon:
        return random.choice(ACTIONS)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        q_values = dqn(state_tensor)
        return ACTIONS[torch.argmax(q_values).item()]

# Function to execute the action
def execute_action(state, action):
    if action == 'expand_line':
        return expand_line(state)
    elif action == 'delete_line':
        return delete_line(state)
    elif action == 'do_nothing':
        return state, 0, False  # No change to state, zero reward, not done

# Example function to expand a line (stub)
def expand_line(state: np.array):
    # Logic to expand a line between two stations
    # For simplicity, assume state is updated and reward is calculated
    next_state = state  # Updated state after expanding line
    reward = 1  # Example reward
    done = False  # Example end condition
    return next_state, reward, done

# Example function to delete a line (stub)
def delete_line(state: np.array):
    
    return next_state, reward, done
