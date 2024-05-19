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

# def make_station_index(stations)->dict:
#     station_index = {station: idx for idx, station in enumerate(stations)}
#     return station_index

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

def get_connected_stations_plus(information, added_path, added_station)->np.array:
    #10 stations,10 stations
    matrix = np.zeros((10, 10), dtype=int)
    station_index = make_station_index(information)
    
    # add station to path
    if added_path in information['paths']:
        information['paths'][added_path].append(added_station.id)
    else:
        information['paths'][added_path] = added_station.id
        
    for color, stations in information['paths'].items():
        if isinstance(stations, list) and len(stations) == 2:
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

#starting station is not considered in this function
def calculate_score(information,path,start_station,connected_station)->int:
    #calculate new connected matrix, 
    adj_matrix= get_connected_stations_plus(information,path,connected_station)
    matrix = get_station_shapes(information)
    for idx,row in enumerate(adj_matrix):
        for y, value in enumerate(row):
            if value == 1:
                matrix[idx] += matrix[y]

    matrix = scaling_down(matrix)
    #above would generate 10*4 matrix, that m is station,n is connected shapes

    passenger_matrix = get_passengers_on_station(information)
    score = 0
    for idx,row in enumerate(passenger_matrix):
        for y, passenger in enumerate(row):
            if matrix[idx][y] >0:
                score += passenger
        
    return score




def get_connected_shape(information)->np.array:
    #10 stations, 4 shapes
    matrix = get_station_shapes(information)
    adj_matrix = get_connected_stations(information)
    
    for idx,row in enumerate(adj_matrix):
        for y, value in enumerate(row):
            if value == 1:
                matrix[idx] += matrix[y]
    return scaling_down(matrix)

#if any element in matrix is bigger than 1, change it down to 1
def scaling_down(matrix):
    for idx,row in enumerate(matrix):
        for y, value in enumerate(row):
            if value > 1:
                matrix[idx][y] = 1
    return matrix
                



