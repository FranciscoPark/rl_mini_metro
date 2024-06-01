import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#actor chooses the action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        return action_probs
    
#critic evaluates the state
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        state_value = self.fc3(x)
        return state_value
    
class A3C:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state)
        action = np.random.choice(np.arange(len(action_probs[0])), p=action_probs[0].detach().numpy())
        return action
    
    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action_probs = self.actor(state)
        action_prob = action_probs[0][action]
        next_action_probs = self.actor(next_state)
        next_action_prob = next_action_probs[0][self.get_action(next_state)]
        
        state_value = self.critic(state)
        next_state_value = self.critic(next_state)
        
        target = reward + self.gamma * next_state_value * (1-done)
        advantage = target - state_value
        
        actor_loss = -torch.log(action_prob) * advantage.detach()
        critic_loss = (target - state_value) ** 2
        
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()