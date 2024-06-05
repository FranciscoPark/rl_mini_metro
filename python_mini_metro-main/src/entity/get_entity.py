from typing import List

from config import screen_height, screen_width
from entity.metro import Metro
from entity.station import Station
from utils import get_random_position, get_random_station_shape
from geometry.utils import distance
import random
import numpy as np

def get_random_station() -> Station:
    shape = get_random_station_shape()
    position = get_random_position(screen_width, screen_height)
    return Station(shape, position)


def get_random_stations(num: int, seed=1) -> List[Station]:
    random.seed(seed)
    np.random.seed(seed)
    stations: List[Station] = []
    for _ in range(num):
        #changed so no overlapping positions
        new_station = get_random_station()
        if len(stations) == 0:
            stations.append(new_station)
            continue
        while any(distance(new_station.position, station.position) < 50 for station in stations):
            new_station = get_random_station()
            
        stations.append(new_station)
    return stations


def get_metros(num: int) -> List[Metro]:
    metros: List[Metro] = []
    for _ in range(num):
        metros.append(Metro())
    return metros
