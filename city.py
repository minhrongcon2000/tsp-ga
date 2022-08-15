from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
from python_tsp.distances import euclidean_distance_matrix
from python_tsp.exact import solve_tsp_dynamic_programming


class Country:
    def __init__(self, num_cities=10):
        self.num_cities = num_cities
        self.cities = self._init_cities_map(num_cities)
        
    def _init_cities_map(self, num_cities):
        return np.random.random((num_cities, 2))
    
    def get_distance(self, city_1, city_2):
        return np.linalg.norm(self.cities[city_1] - self.cities[city_2])
    
    def total_distance(self, path):
        s = 0
        for i in range(len(path) - 1):
            s += self.get_distance(path[i], path[i + 1])
        s += self.get_distance(path[-1], path[0])
        return s
    
    def show_map(self):
        plt.scatter(self.cities[:, 0], self.cities[:, 1])
        plt.show()
        
    def get_tsp_solution(self):
        distance_matrix = euclidean_distance_matrix(self.cities)
        return solve_tsp_dynamic_programming(distance_matrix)