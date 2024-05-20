from __future__ import annotations

import pprint
import random
from typing import Dict, List
from rl_agent import Agent
import pygame

import numpy as np

from config import (
    num_metros,
    num_paths,
    num_stations,
    passenger_color,
    passenger_size,
    passenger_spawning_interval_step,
    passenger_spawning_start_step,
    score_display_coords,
    score_font_size,
    gameover_font_size,
    gameover_text,
    gameover_text_color,
    gameover_text_coords,
    gameover_text_center,
    start_with_3_initial_paths,
    station_shape_type_list,
    greedy_agent
)
from entity.get_entity import get_random_stations
from entity.metro import Metro
from entity.passenger import Passenger
from entity.path import Path
from entity.station import Station
from event.event import Event
from event.keyboard import KeyboardEvent
from event.mouse import MouseEvent
from event.type import KeyboardEventType, MouseEventType
from geometry.point import Point
from geometry.type import ShapeType
from graph.graph_algo import bfs, build_station_nodes_dict
from graph.node import Node
from travel_plan import TravelPlan
from type import Color
from ui.button import Button
from ui.path_button import PathButton, get_path_buttons
from utils import get_shape_from_type, hue_to_rgb


TravelPlans = Dict[Passenger, TravelPlan]
pp = pprint.PrettyPrinter(indent=4)


class Mediator:
    def __init__(self) -> None:
        pygame.font.init()

        # set random seed
        np.random.seed(42)
        random.seed(42)
        
        # configs
        self.passenger_spawning_step = passenger_spawning_start_step
        self.passenger_spawning_interval_step = passenger_spawning_interval_step
        self.num_paths = num_paths
        self.num_metros = num_metros
        self.num_stations = num_stations
        self.start_with_3_initial_paths = start_with_3_initial_paths

        # UI
        self.path_buttons = get_path_buttons(self.num_paths)
        self.path_to_button: Dict[Path, PathButton] = {}
        self.buttons = [*self.path_buttons]
        self.font = pygame.font.SysFont("arial", score_font_size)

        #gameover UI
        self.gameover_text = gameover_text
        self.gameover_text_color = gameover_text_color
        self.gameover_text_coords = gameover_text_coords
        self.gameover_text_center = gameover_text_center
        self.gameover_font_size = gameover_font_size
        self.gameover = False



        # entities
        self.stations = get_random_stations(self.num_stations)
        self.metros: List[Metro] = []
        self.paths: List[Path] = []
        self.passengers: List[Passenger] = []
        self.path_colors: Dict[Color, bool] = {}
        
        for i in range(num_paths):
            color = hue_to_rgb(i / (num_paths + 1))
            self.path_colors[color] = False  # not taken
        self.path_to_color: Dict[Path, Color] = {}
    
        self.color_name: Dict[Color, str] = {(255.0, 0.0, 0.0): 'red', (0.0, 255.0, 255.0): 'blue', (127.5, 255.0, 0.0): 'green'}
        
        

        # status
        self.time_ms = 0
        self.steps = 0
        self.steps_since_last_spawn = self.passenger_spawning_interval_step + 1
        self.is_mouse_down = False
        self.is_creating_path = False
        self.path_being_created: Path | None = None
        self.travel_plans: TravelPlans = {}
        self.is_paused = False
        self.score = 0

    #gameover
    def display_gameover(self, screen: pygame.surface.Surface) -> None:
        gameover_font = pygame.font.SysFont("arial", self.gameover_font_size)
        gameover_text = gameover_font.render(self.gameover_text, True, self.gameover_text_color)
        screen.blit(gameover_text, gameover_text_coords)
    
    def is_gameover(self):
        return not self.gameover
    
    #initialize with 3 paths
    def initialize_with_3_paths(self):
        #randomly choose 3 stations
        stations = random.sample(self.stations, 3)
        for station in stations:
            self.start_path_on_station(station)
            #randomly choose 1 to connect
            station_to_connect = random.choice([x for x in self.stations if x not in stations])
            self.add_station_to_path(station_to_connect)
            self.end_path_on_station(station_to_connect)
        
    def assign_paths_to_buttons(self):
        for path_button in self.path_buttons:
            path_button.remove_path()

        self.path_to_button = {}
        for i in range(min(len(self.paths), len(self.path_buttons))):
            path = self.paths[i]
            button = self.path_buttons[i]
            button.assign_path(path)
            self.path_to_button[path] = button

    def render(self, screen: pygame.surface.Surface) -> None:
        if self.gameover == True:
            self.display_gameover(screen)
            return
        for idx, path in enumerate(self.paths):
            path_order = idx - round(self.num_paths / 2)
            path.draw(screen, path_order)
        for station in self.stations:
            station.draw(screen)
        for metro in self.metros:
            metro.draw(screen)
        for button in self.buttons:
            button.draw(screen)
            
        text_surface = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
        screen.blit(text_surface, score_display_coords)
        

    def react_mouse_event(self, event: MouseEvent):
        entity = self.get_containing_entity(event.position)

        if event.event_type == MouseEventType.MOUSE_DOWN:
            self.is_mouse_down = True
            if entity:
                if isinstance(entity, Station):
                    self.start_path_on_station(entity)

        elif event.event_type == MouseEventType.MOUSE_UP:
            self.is_mouse_down = False
            if self.is_creating_path:
                assert self.path_being_created is not None
                if entity and isinstance(entity, Station):
                    self.end_path_on_station(entity)
                else:
                    self.abort_path_creation()
            else:
                if entity and isinstance(entity, PathButton):
                    if entity.path:
                        self.remove_path(entity.path)

        elif event.event_type == MouseEventType.MOUSE_MOTION:
            if self.is_mouse_down:
                if self.is_creating_path and self.path_being_created:
                    if entity and isinstance(entity, Station):
                        self.add_station_to_path(entity)
                    else:
                        self.path_being_created.set_temporary_point(event.position)
            else:
                if entity and isinstance(entity, Button):
                    entity.on_hover()
                else:
                    for button in self.buttons:
                        button.on_exit()

    def react_keyboard_event(self, event: KeyboardEvent):
        if event.event_type == KeyboardEventType.KEY_UP:
            if event.key == pygame.K_SPACE:
                self.is_paused = not self.is_paused

    def react(self, event: Event | None):
        if isinstance(event, MouseEvent):
            self.react_mouse_event(event)
        elif isinstance(event, KeyboardEvent):
            self.react_keyboard_event(event)

    def get_containing_entity(self, position: Point):
        for station in self.stations:
            if station.contains(position):
                return station
        for button in self.buttons:
            if button.contains(position):
                return button

    def remove_path(self, path: Path):
        self.path_to_button[path].remove_path()
        for metro in path.metros:
            for passenger in metro.passengers:
                self.passengers.remove(passenger)
            self.metros.remove(metro)
        self.release_color_for_path(path)
        self.paths.remove(path)
        self.assign_paths_to_buttons()
        self.find_travel_plan_for_passengers()

    def start_path_on_station(self, station: Station) -> None:
        if len(self.paths) < self.num_paths:
            self.is_creating_path = True
            assigned_color = (0, 0, 0)
            for path_color, taken in self.path_colors.items():
                if not taken:
                    assigned_color = path_color
                    self.path_colors[path_color] = True
                    break
            path = Path(assigned_color)
            self.path_to_color[path] = assigned_color
            path.add_station(station)
            path.is_being_created = True
            self.path_being_created = path
            self.paths.append(path)

    def add_station_to_path(self, station: Station) -> None:
        assert self.path_being_created is not None
        if self.path_being_created.stations[-1] == station:
            return
        # loop
        if (
            len(self.path_being_created.stations) > 1
            and self.path_being_created.stations[0] == station
        ):
            self.path_being_created.set_loop()
        # non-loop
        elif self.path_being_created.stations[0] != station:
            if self.path_being_created.is_looped:
                self.path_being_created.remove_loop()
            self.path_being_created.add_station(station)

    def abort_path_creation(self) -> None:
        assert self.path_being_created is not None
        self.is_creating_path = False
        self.release_color_for_path(self.path_being_created)
        self.paths.remove(self.path_being_created)
        self.path_being_created = None

    def release_color_for_path(self, path: Path) -> None:
        self.path_colors[path.color] = False
        del self.path_to_color[path]

    def finish_path_creation(self) -> None:
        assert self.path_being_created is not None
        self.is_creating_path = False
        self.path_being_created.is_being_created = False
        self.path_being_created.remove_temporary_point()
        if len(self.metros) < self.num_metros:
            metro = Metro()
            self.path_being_created.add_metro(metro)
            self.metros.append(metro)
        self.path_being_created = None
        self.assign_paths_to_buttons()

    def end_path_on_station(self, station: Station) -> None:
        assert self.path_being_created is not None
        # current station de-dupe
        if (
            len(self.path_being_created.stations) > 1
            and self.path_being_created.stations[-1] == station
        ):
            self.finish_path_creation()
        # loop
        elif (
            len(self.path_being_created.stations) > 1
            and self.path_being_created.stations[0] == station
        ):
            self.path_being_created.set_loop()
            self.finish_path_creation()
        # non-loop
        elif self.path_being_created.stations[0] != station:
            self.path_being_created.add_station(station)
            self.finish_path_creation()
        else:
            self.abort_path_creation()

    def get_station_shape_types(self):
        station_shape_types: List[ShapeType] = []
        for station in self.stations:
            if station.shape.type not in station_shape_types:
                station_shape_types.append(station.shape.type)
        return station_shape_types

    def is_passenger_spawn_time(self) -> bool:
        return (
            self.steps == self.passenger_spawning_step
            or self.steps_since_last_spawn == self.passenger_spawning_interval_step
        )

    def spawn_passengers(self):
        #gameover
        for station in self.stations:
            station_types = self.get_station_shape_types()
            other_station_shape_types = [
                x for x in station_types if x != station.shape.type
            ]
            destination_shape_type = random.choice(other_station_shape_types)
            destination_shape = get_shape_from_type(
                destination_shape_type, passenger_color, passenger_size
            )
            passenger = Passenger(destination_shape)
            if station.has_room():
                station.add_passenger(passenger)
                self.passengers.append(passenger)
            else:
                #gameover
                self.gameover = True
                break
                


    def increment_time(self, dt_ms: int) -> None:
        if self.is_paused:
            return

        # record time
        self.time_ms += dt_ms
        self.steps += 1
        self.steps_since_last_spawn += 1

        # move metros
        for path in self.paths:
            for metro in path.metros:
                path.move_metro(metro, dt_ms)

        # spawn passengers
        if self.is_passenger_spawn_time():
            self.spawn_passengers()
            #if gameover
            if self.gameover == True:
                return
            self.steps_since_last_spawn = 0
        
        self.find_travel_plan_for_passengers()
        self.move_passengers()
        #print(self.save_state())
        
        
        #greedy agent
        if greedy_agent:
            #steps for graphical display, steps for agent to choose action
            if self.steps%1000 == 10:
                state = self.save_state()
                agent = Agent(state, 0) # input state and Exploration rate
                #agent.print_state()
                action = agent.choose_action()
                #print(self.path_to_color)
                #print(action)
                #self.agent_add_station_to_path(action[0],action[1])
                # action = agent.choose_greedy_action()
                self.agent_add_station_to_path(action[0],action[1],action[2])



    def move_passengers(self) -> None:
        for metro in self.metros:
            if metro.current_station:
                passengers_to_remove = []
                passengers_from_metro_to_station = []
                passengers_from_station_to_metro = []

                # queue
                for passenger in metro.passengers:
                    if (
                        metro.current_station.shape.type
                        == passenger.destination_shape.type
                    ):
                        passengers_to_remove.append(passenger)
                    elif (
                        self.travel_plans[passenger].get_next_station()
                        == metro.current_station
                    ):
                        passengers_from_metro_to_station.append(passenger)
                for passenger in metro.current_station.passengers:
                    if (
                        self.travel_plans[passenger].next_path
                        and self.travel_plans[passenger].next_path.id == metro.path_id  # type: ignore
                    ):
                        passengers_from_station_to_metro.append(passenger)

                # process
                for passenger in passengers_to_remove:
                    passenger.is_at_destination = True
                    metro.remove_passenger(passenger)
                    self.passengers.remove(passenger)
                    del self.travel_plans[passenger]
                    self.score += 1

                for passenger in passengers_from_metro_to_station:
                    if metro.current_station.has_room():
                        metro.move_passenger(passenger, metro.current_station)
                        self.travel_plans[passenger].increment_next_station()
                        self.find_next_path_for_passenger_at_station(
                            passenger, metro.current_station
                        )

                for passenger in passengers_from_station_to_metro:
                    if metro.has_room():
                        metro.current_station.move_passenger(passenger, metro)

    def get_stations_for_shape_type(self, shape_type: ShapeType):
        stations: List[Station] = []
        for station in self.stations:
            if station.shape.type == shape_type:
                stations.append(station)
        random.shuffle(stations)

        return stations

    def find_shared_path(self, station_a: Station, station_b: Station) -> Path | None:
        for path in self.paths:
            stations = path.stations
            if (station_a in stations) and (station_b in stations):
                return path
        return None

    def passenger_has_travel_plan(self, passenger: Passenger) -> bool:
        return (
            passenger in self.travel_plans
            and self.travel_plans[passenger].next_path is not None
        )

    def find_next_path_for_passenger_at_station(
        self, passenger: Passenger, station: Station
    ):
        next_station = self.travel_plans[passenger].get_next_station()
        assert next_station is not None
        next_path = self.find_shared_path(station, next_station)
        self.travel_plans[passenger].next_path = next_path

    def skip_stations_on_same_path(self, node_path: List[Node]):
        assert len(node_path) >= 2
        if len(node_path) == 2:
            return node_path
        else:
            nodes_to_remove = []
            i = 0
            j = 1
            path_set_list = [x.paths for x in node_path]
            path_set_list.append(set())
            while j <= len(path_set_list) - 1:
                set_a = path_set_list[i]
                set_b = path_set_list[j]
                if set_a & set_b:
                    j += 1
                else:
                    for k in range(i + 1, j - 1):
                        nodes_to_remove.append(node_path[k])
                    i = j - 1
                    j += 1
            for node in nodes_to_remove:
                node_path.remove(node)
        return node_path

    def find_travel_plan_for_passengers(self) -> None:
        station_nodes_dict = build_station_nodes_dict(self.stations, self.paths)
        for station in self.stations:
            for passenger in station.passengers:
                if not self.passenger_has_travel_plan(passenger):
                    possible_dst_stations = self.get_stations_for_shape_type(
                        passenger.destination_shape.type
                    )
                    should_set_null_path = True
                    for possible_dst_station in possible_dst_stations:
                        start = station_nodes_dict[station]
                        end = station_nodes_dict[possible_dst_station]
                        node_path = bfs(start, end)
                        if len(node_path) == 1:
                            # passenger arrived at destination
                            station.remove_passenger(passenger)
                            self.passengers.remove(passenger)
                            passenger.is_at_destination = True
                            del self.travel_plans[passenger]
                            should_set_null_path = False
                            break
                        elif len(node_path) > 1:
                            node_path = self.skip_stations_on_same_path(node_path)
                            self.travel_plans[passenger] = TravelPlan(node_path[1:])
                            self.find_next_path_for_passenger_at_station(
                                passenger, station
                            )
                            should_set_null_path = False
                            break
                    if should_set_null_path:
                        self.travel_plans[passenger] = TravelPlan([])


    def count_passengers_by_type(self, station: Station) -> Dict:
        station_shape_type_list_str = [str(shape_type) for shape_type in station_shape_type_list]
        count = dict(zip(station_shape_type_list_str, [0] * len(station_shape_type_list)))
        for passenger in station.passengers:
            shape = str(passenger.destination_shape.type)
            if shape in count:
                count[shape] += 1
        return count
    

    
    def save_state(self) -> Dict:
        state = {
            'step': self.steps,
            'num_stations': self.num_stations,
            'stations': [station for station in self.stations],
            'station_ids': [station.id for station in self.stations],
            'station_shapes': [str(station.shape.type) for station in self.stations],
            'station_passengers': {station.id: self.count_passengers_by_type(station) for station in self.stations},
            'paths': {path : [station.id for station in path.stations] for path in self.paths},
            'paths_colorname': {self.color_name[path.color] : [station.id for station in path.stations] for path in self.paths},
            'path_color': [path.color for path in self.paths],
            'paths_adj_matrix': {self.color_name[path.color]: self.adjacency_matrix(path) for path in self.paths},
            'score': self.score 
        }
        return state
    


    
    def adjacency_matrix(self, path: Path):
        """ Creates an adjacency matrix for a given path"""

        # initiate matrix with zeros for all stations in game
        adjacency_matrix = [[0] * len(self.stations) for _ in range(len(self.stations))]
        station_id_to_index = {station.id: i for i, station in enumerate(self.stations)}
        
        # fill in adjacency matrix 
        for i in range(len(path.stations) - 1):
            start_station = path.stations[i]
            next_station = path.stations[i + 1]
            start_index = station_id_to_index[start_station.id]
            next_index = station_id_to_index[next_station.id]
            adjacency_matrix[start_index][next_index] = 1
            adjacency_matrix[next_index][start_index] = 1
        return adjacency_matrix




    def agent_add_station_to_path(self, path: Path, station_to_add: Station, add_last=True) -> None:
        # delete path
        if path in self.path_to_button:
            self.remove_path(path)

        # add station to path
        if add_last:
            self.start_path_on_station(path.stations[0])
            for station in path.stations[1:]:
                self.add_station_to_path(station)
            self.end_path_on_station(station_to_add)

        else: # add first
            self.start_path_on_station(station_to_add)
            for station in path.stations[:-1]:
                self.add_station_to_path(station)
            self.end_path_on_station(path.stations[-1])
    
    
    def available_actions(self):
        """ Returns the available actions for the current state"""
        pass