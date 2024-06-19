import numpy as np

class State:
    def __init__(self,state):
        self.state =state
        self.shape_order = ['ShapeType.RECT', 'ShapeType.CIRCLE', 'ShapeType.TRIANGLE', 'ShapeType.CROSS']
        self.station_passengers = self.state['station_passengers']
        self.station_shapes = self.state['station_shapes']
        self.paths = self.state['paths']
        self.station_index = self.make_station_index()
        
        self.passengers_on_station= self.get_passengers_on_station()
        
        self.station_shapes_matrix = self.get_station_shapes()
        self.connected_stations = self.get_connected_stations()
        
    def make_station_index(self)->dict:
        station_index = {station: idx for idx, station in enumerate(self.station_passengers.keys())}
        return station_index
    def get_passengers_on_station(self)->np.array:
        #10 stations,4 shapes for now
        matrix = np.zeros((10, 4), dtype=int)
        for i, (station, shapes) in enumerate(self.station_passengers.items()):
            matrix[i] = [shapes['ShapeType.RECT'], shapes['ShapeType.CIRCLE'], shapes['ShapeType.TRIANGLE'], shapes['ShapeType.CROSS']]
        return matrix
    def get_station_shapes(self)->np.array:
        #10 stations,4 shapes for now, one-hot encoding
        matrix = np.zeros((10, 4), dtype=int)
        #print(information['station_shapes'])
        for i, station in enumerate(self.station_shapes):
            matrix[i,self.shape_order.index(station)] = 1
        return matrix
    def get_connected_stations(self)->np.array:
        #10 stations,10 stations
        matrix = np.zeros((10, 10), dtype=int)
        station_index = self.make_station_index()

        for color, stations in self.paths.items():
            if len(stations) >= 2:
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
    def get_connected_stations_plus(self, added_path, added_station)->np.array:
        #10 stations,10 stations
        matrix = np.zeros((10, 10), dtype=int)
        station_index = self.make_station_index()
        
        # add station to path
        if added_path in self.state['paths']:
            self.state['paths'][added_path].append(added_station.id)
        else:
            self.state['paths'][added_path] = added_station.id
            
        for color, stations in self.state['paths'].items():
            if isinstance(stations, list) and len(stations) >= 2:
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

    def calculate_score(self,path,start_station,connected_station)->int:
        adj_matrix= self.get_connected_stations_plus(path,connected_station)
        matrix = self.station_shapes_matrix
        for idx,row in enumerate(adj_matrix):
            for y, value in enumerate(row):
                if value == 1:
                    matrix[idx] += matrix[y]
        
        matrix = self.scaling_down(matrix)
        passenger_matrix = self.passengers_on_station
        score = 0
        for idx,row in enumerate(passenger_matrix):
            for y, passenger in enumerate(row):
                if matrix[idx][y] >0:
                    score += passenger

        return score

    def scaling_down(self,matrix)->np.array:
        for idx,row in enumerate(matrix):
            for y, value in enumerate(row):
                if value > 1:
                    matrix[idx][y] = 1
        return matrix
    
    def get_connected_shape(self)->np.array:
        #10 stations, 4 shapes
        matrix =  self.station_shapes_matrix
        adj_matrix = self.connected_stations
        
        for idx,row in enumerate(adj_matrix):
            for y, value in enumerate(row):
                if value == 1:
                    matrix[idx] += matrix[y]
        return self.scaling_down(matrix)
    
    def count_station_in_path(self, target_station_id)->int:
        count = 0
        for station_ids in self.paths.values():
            for station_id in station_ids:
                if station_id == target_station_id:
                    count += 1
        
        return count



















