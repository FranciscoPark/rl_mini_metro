
import pygame
import numpy as np


from state import *
import random

from entity.passenger import Passenger
from entity.path import Path
from entity.station import Station



class Agent:
    def __init__(self, state, epsilon=0.1):
        self.state = state
        # entities
        self.stations = self.state['stations']
        self.paths = self.state['paths'] 
        
        self.station_passengers = self.state['station_passengers']
        self.station_shapes = self.state['station_shapes']
        self.paths_adj_matrix = self.state['paths_adj_matrix'] # only direct connection
        
        # Q-table, state, and action space
        self.action_space = self.define_action_space()
        self.q_table = {action: 0 for action in self.action_space}

        self.epsilon = epsilon  # Exploration rate

    #testing new function    
    # def print_state(self):
    #     print(get_connected_shape(self.state))
    
    def define_action_space(self):
        actions = []
        for station in self.stations:
            for path in self.paths.keys():
                # exclude stations already in the path ?
                if station.id not in self.paths[path]:
                    actions.append((path, station)) # action to draw line to a station using a path 
        return actions
    
    def compute_possible_deliveries(self):
        delivery_count = {action: 0 for action in self.action_space}
        
        for action_idx, (path, station) in enumerate(self.action_space):
            station_idx = action_idx//len(self.paths)
            
            # for passenger in station.passengers:
            connected_matrix = get_connected_stations_plus(self.state, path, station)
            connected_idcs = np.argwhere(connected_matrix==1)
            
            for connected_idx in connected_idcs:
                station_from = self.stations[connected_idx[0]]
                station_to_idx = connected_idx[1]
                delivery_count[(path, station)] += self.station_passengers[station_from.id][self.station_shapes[station_to_idx]]

            self.paths[path].remove(station.id)
        
        return delivery_count

    def choose_action(self): 
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return random.choice(self.action_space)
        else:
            # Exploit: choose the action with the highest Q-value
            self.q_table = self.compute_possible_deliveries()
            return max(self.q_table, key=self.q_table.get)


    # def auto_decision_maker(self):
    #     # Choose an action
    #     action = self.choose_action()
        
    #     # Execute the action and get the next state
    #     mediator.agent_add_station_to_path(action[0],action[1])
        
    #     # will add more options for action
    #     if action == "modify_connection":
    #         self.modify_connection()
    #     elif action == "delete_connection":
    #         self.delete_connection()
    #     elif action == "observe":
    #         pass  # Do nothing