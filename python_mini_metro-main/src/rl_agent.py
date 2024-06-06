
import pygame
import numpy as np


from state import State
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
        self.paths_with_colors = self.state['paths_colorname']
        
        self.station_passengers = self.state['station_passengers']
        self.station_shapes = self.state['station_shapes']
        self.paths_adj_matrix = self.state['paths_adj_matrix'] # only direct connection
        
        # Q-table, state, and action space
        self.action_space = self.define_action_space()
        self.q_table = {action: 0 for action in self.action_space}

        self.epsilon = epsilon  # Exploration rate
        self.action_cnt =0
        #class instance
        self.State = State(self.state)

    #testing new function    
    # def print_state(self):
    #     print(self.state['paths'])
    #     print(get_connected_stations(self.state))
        
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
            connected_matrix = self.State.get_connected_stations_plus(path, station)
            connected_idcs = np.argwhere(connected_matrix==1)
            
            for connected_idx in connected_idcs:
                station_from = self.stations[connected_idx[0]]
                station_to_idx = connected_idx[1]
                delivery_count[(path, station)] += self.station_passengers[station_from.id][self.station_shapes[station_to_idx]]

            self.paths[path].remove(station.id)
        
        return delivery_count
    
    def compute_maximum_delivery_route(self)->list:
        results = []
        weights = {}
        
        
        # calculate weight vector for each station
        for station in self.stations:
            
            weights[station.id] = 1/(self.State.count_station_in_path(station.id)+1e-10)
        #print(weights)
        
        # for each color, calculate possible connections
        for path, stations in self.paths.items():
            if len(stations) > 1:
                first_station = stations[0]
                last_station = stations[-1]
                for station in self.stations:
                    #has to be a station.id
                    if station.id not in stations:
                        add_on_first= self.calculate_delivery_score(path,first_station, station)
                        add_on_last= self.calculate_delivery_score(path,last_station,station)
                        results.append({
                        'score': add_on_first,
                        'weighted_score': add_on_first * weights[station.id],
                        'path': path,
                        'start_station': first_station,
                        'connected_station': station,
                        'add_last': False
                    })
                        results.append({
                            'score': add_on_last,
                            'weighted_score': add_on_last * weights[station.id],
                            'path': path,
                            'start_station': last_station,
                            'connected_station': station,
                            'add_last': True
                        })
        return results
    
    def get_maximum_delivery_route(self)->dict:        
        results = self.compute_maximum_delivery_route()
        #{'score': 15, 'path': 'path_a', 'start_station': 'Station-C', 
        #'connected_station': 'Station-D'}
        # return max(results, key=lambda x: x['score'])
        return max(results, key=lambda x: x['weighted_score'])

    def calculate_delivery_score(self,path, start_station, connected_station)->int:
        #connected on last or first is not considered here.
        return self.State.calculate_score(path, start_station, connected_station)
    
    def choose_greedy_action(self)->tuple:
        results = self.get_maximum_delivery_route()
        # print(results)
        return results['path'], results['connected_station'], results['add_last']

    def choose_action(self): 
        #dont do anything after 24 actions
        if self.action_cnt == 24:
            return
        self.action_cnt += 1
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            action = random.choice(self.action_space)
            return action[0], action[1], random.choice([True,False])
        else:
            # Exploit: choose the action with the highest Q-value
            # self.q_table = self.compute_possible_deliveries()
            # return max(self.q_table, key=self.q_table.get)
            return self.choose_greedy_action()

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