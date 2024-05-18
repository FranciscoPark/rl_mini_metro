
import pygame

from mediator import Mediator
import random

class Agent:
    def __init__(self, num_paths, stations, paths):
        # Initialize Q-table, state, and action space
        self.q_table = {}
        # self.state_space = self.define_state_space()
        self.action_space = self.define_action_space()
        self.epsilon = 0.1  # Exploration rate
        # self.alpha = 0.1  # Learning rate
        # self.gamma = 0.9  # Discount factor
        
    def define_action_space(self):
        actions = []
        for path in self.num_paths:
            for station in self.stations:
                # exclude current station / already connected stations ??
                actions.append((path, station)) # draw line to a station using a path 
        return actions
    
    def compute_possible_deliveries(self):
        delivery_count = {'action': 0 for action in self.action_space}
        
        for path, station in self.action_space:
            # possible path that added station to current path.station (not actually update the path)
            new_paths = self.paths.copy() 
            new_paths.add_station_to_path(station)
            new_paths.end_path_on_station(station)
            
            for passenger in station.passengers:
                for target_station in self.stations:                    
                    if self.is_connected(station, target_station, new_paths) and passenger.destination == target_station.shape.type:
                        delivery_count[(path, station)] += 1
                        break # to prevent counting twice for the same passenger
        
        return delivery_count

    def is_connected(self, station_a, station_b, updated_paths):        
        for path in updated_paths:
            # adj_mat = adjacency_matrix(path) 
            # if adj_mat[a_idx][b_idx]==1:
                # return True
            # both station in the same path
            if station_a in path.stations and station_b in path.stations:
                    return True
        return False
    
    def choose_action(self): # single state: current station agent is in
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return random.choice(self.action_space)
        else:
            # Exploit: choose the action with the highest Q-value
            self.q_table = self.compute_possible_deliveries()
            return max(self.q_table, key=self.q_table.get)

    def auto_decision_maker(self):
        # Choose an action
        action = self.choose_action()
        
        # Execute the action and get the next state
        action[0].add_station(action[1])

        ## will add more options for action
        # if action == "modify_connection":
        #     self.modify_connection()
        # elif action == "delete_connection":
        #     self.delete_connection()
        # elif action == "observe":
        #     pass  # Do nothing

# fuse to mediator.react() function
agent = Agent(num_paths, stations, paths)
agent.auto_decision_maker()