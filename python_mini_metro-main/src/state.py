import numpy as np

def get_passengers_on_station(information)->np.array:
    #10 stations,4 shapes for now
    matrix = np.zeros((10, 4), dtype=int)
    for i, (station, shapes) in enumerate(information['station_passengers'].items()):
        matrix[i] = [shapes['ShapeType.RECT'], shapes['ShapeType.CIRCLE'], shapes['ShapeType.TRIANGLE'], shapes['ShapeType.CROSS']]
    return matrix

def get_station_shapes(information)->np.array:
    #10 stations,4 shapes for now, one-hot encoding
    matrix = np.zeros((10, 4), dtype=int)
    shape_order = ['ShapeType.RECT', 'ShapeType.CIRCLE', 'ShapeType.TRIANGLE', 'ShapeType.CROSS']
    #print(information['station_shapes'])
    for i, station in enumerate(information['station_shapes']):
        matrix[i, shape_order.index(station)] = 1
    return matrix

def make_station_index(information)->dict:
    station_index = {station: idx for idx, station in enumerate(information['station_passengers'].keys())}
    return station_index

def get_connected_stations(information)->np.array:
    #10 stations,10 stations
    matrix = np.zeros((10, 10), dtype=int)
    station_index = make_station_index(information)
    for color, stations in information['paths'].items():
        if len(stations) == 2:
            idx1 = station_index[stations[0]]
            idx2 = station_index[stations[1]]
            matrix[idx1, idx2] = 1
            matrix[idx2, idx1] = 1

    #check mutual connection
    for idx,row in enumerate(matrix):
       for y, value in enumerate(row):
           if value == 1:
              for t, val in enumerate(matrix[y]):
                    if val == 1 and t != idx:
                        matrix[idx][t] = 1
                  
    return matrix



    