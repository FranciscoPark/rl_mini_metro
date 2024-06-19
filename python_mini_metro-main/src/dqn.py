import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

# normal pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# graph pytorch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm import GraphNorm


Transition = namedtuple('Transition',
                        ('state', 'mask', 'action', 'next_state', 'next_mask', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    



class DQL_GCN(torch.nn.Module):
    def __init__(self, num_node_features, n_actions, num_stations):
        super().__init__()
        self.num_node_features = num_node_features
        self.n_actions = n_actions
        self.num_stations = num_stations

        #LAYERS
        self.conv1 = GCNConv(num_node_features, 4)
        self.conv2 = GCNConv(4, 1)
        self.ff1 = torch.nn.Linear(num_stations*3, 64)
        self.ff2 = torch.nn.Linear(64, n_actions)
        self.norm = GraphNorm(num_node_features)

    def forward(self, red_graph, blue_graph, green_graph, mask, batch_size=5):
        # run convolution on each graph
        x_red, edge_index_red = self.norm(red_graph.x), red_graph.edge_index
        x_blue, edge_index_blue = self.norm(blue_graph.x), blue_graph.edge_index
        x_green, edge_index_green = self.norm(green_graph.x), green_graph.edge_index
        
        # first convolution
        x_red = F.relu(self.conv1(x_red, edge_index_red))#.reshape(5, self.num_node_features)     # 5 in batch, X features
        x_blue = F.relu(self.conv1(x_blue, edge_index_blue))#.reshape(5, self.num_node_features) 
        x_green = F.relu(self.conv1(x_green, edge_index_green))#.reshape(5, self.num_node_features) 

        # second convolution
        x_red = F.relu(self.conv2(x_red, edge_index_red))
        x_blue = F.relu(self.conv2(x_blue, edge_index_blue))
        x_green = F.relu(self.conv2(x_green, edge_index_green))
        

        # reshape according to batch size
        x_red = x_red.reshape(batch_size, self.num_stations)
        x_blue = x_blue.reshape(batch_size, self.num_stations)
        x_green = x_green.reshape(batch_size, self.num_stations)


        # combine the three outputs by appending them to a single vector
        x = torch.cat((x_red, x_blue, x_green), dim=1)

        # feed through feedforward layers
        x = F.relu(self.ff1(x))
        output = self.ff2(x)

        # mask the output
        mask = mask.reshape(batch_size, self.n_actions)
        output = output * mask

        return output