from itertools import permutations
from city import Country
import numpy as np
import heapq
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from python_tsp.distances import euclidean_distance_matrix
from python_tsp.exact import solve_tsp_dynamic_programming


class TSPGA:
    def __init__(self, 
                 country: Country, 
                 init_population_size: int=10, 
                 max_population_size: int=100,
                 debug=False):
        """constructor

        Args:
            country (Country): refer to city.Country class
            init_population_size (int, optional): Initial population size when solving TSP. Defaults to 10.
            max_population_size (int, optional): Maximum population size when solving TSP. Defaults to 100.
            debug (bool, optional): Debug mode toggle. Defaults to False.
            Warning: slow for large number of cities should only try with num_cities < 15.
        """
        self.country = country
        self.init_size = init_population_size
        self.max_size = max_population_size
        self.population = self._init_population()
        self.fig, (self.path_ax, self.score_ax) = plt.subplots(2, 1)
        self.debug = debug
        if self.debug:
            _, self.ideal_distance = self.country.get_tsp_solution()
        self.scores = []
        
    def _init_population(self):
        """Initialize random population

        Returns:
            list: a list of population with heap organization. Useful for later operation
        """ 
        init_pop = []
        all_paths = permutations(range(self.country.num_cities))
        
        for _ in range(self.init_size):
            path = next(all_paths)
            score = self.country.total_distance(path)
            heapq.heappush(init_pop, (-score, list(path)))
            
        return init_pop
    
    def add_new_population(self, path):
        """Add new population to the list

        Args:
            path (list[int]): Cyclic path to all cities
        """
        score = self.country.total_distance(path)
        if len(self.population) <= self.max_size:
            heapq.heappush(self.population, (-score, list(path)))
        else:
            # only keep some top population and kill bad population to boost convergence
            heapq.heappushpop(self.population, (-score, list(path)))
            
    def select_best_individual(self, n_best=10):
        """take n-best population based on path length. This is when heap becomes powerful

        Args:
            n_best (int, optional): number of top population to take. Defaults to 10.

        Returns:
            list[list[int]]: top n population
        """
        assert n_best <= len(self.population)
        return heapq.nlargest(n_best, self.population)
    
    def path2adj_list(self, path):
        """generate adjacency list from cyclic path

        Args:
            path (list[int]): cyclic path

        Returns:
            dict[int, set]: adjacency list from given path
        """
        neighbor_list = {}
        for i in range(len(path)):
            neighbor_list[path[i]] = {
                path[(i - 1) % self.country.num_cities], 
                path[(i + 1) % self.country.num_cities]
            }
        return neighbor_list
    
    def merge_neighbor_list(self, neighbor_list1, neighbor_list2):
        """generate a graph that contains all edges from two parent cyclic path

        Args:
            neighbor_list1 (list[int]): parent path 1
            neighbor_list2 (list[int]): parent path 2

        Returns:
            dict[int, set]: merged parent graph
        """
        neighbor_list_merge = {}
        
        for vertex in neighbor_list1:
            neighbor_list_merge[vertex] = neighbor_list1[vertex].union(neighbor_list2[vertex])
            
        return neighbor_list_merge
    
    def breed(self, parent1, parent2):
        """breed parent to create new children

        Args:
            parent1 (list[int]): parent path 1
            parent2 (list[int]): parent path 2

        Returns:
            list[int]: new-born path
        """
        adj_list_1 = self.path2adj_list(parent1)
        adj_list_2 = self.path2adj_list(parent2)
        adj_list_merge = self.merge_neighbor_list(adj_list_1, adj_list_2)
        vertex = random.choice([parent1[0], parent2[0]])
        
        paths = [vertex]
        
        while len(paths) < self.country.num_cities:
            minLen = None
            chosen_vertex = []
            for v in adj_list_merge[vertex]:
                adj_list_merge[v].remove(vertex)
                if minLen is None or len(adj_list_merge[v]) < minLen:
                    minLen = len(adj_list_merge[v])
                    chosen_vertex = [v]
                elif len(adj_list_merge[v]) == minLen:
                    chosen_vertex.append(v)
            adj_list_merge.pop(vertex)
            if len(chosen_vertex) != 0:
                paths.append(random.choice(chosen_vertex))
            else:
                paths.append(random.choice(list(adj_list_merge.keys())))
            vertex = paths[-1]
        return paths
    
    def mutate(self, individual, n_mutate_times=1):
        """mutate individual to mimic the nature of genome

        Args:
            individual (list[int]): a path within population
            n_mutate_times (int, optional): number of times mutation happens. Defaults to 1.

        Returns:
            list[int]: mutated individual
        """
        for _ in range(n_mutate_times):
            gen_pos_1, gen_pos_2 = np.random.choice(self.country.num_cities, 2, replace=False)
            individual[gen_pos_1], individual[gen_pos_2] = individual[gen_pos_2], individual[gen_pos_1]
        return individual
    
    def generate_single_frame(self, 
                              t, 
                              individual_per_epochs=10, 
                              n_breed_per_epoch=10, 
                              mutate_percentage=0.1):
        # select best individuals
        best_individuals = self.select_best_individual(individual_per_epochs)
        
        # breeding
        for _ in range(n_breed_per_epoch):
            (_, individual_1), (_, individual_2) = random.sample(best_individuals, 2)
            new_individual = self.breed(individual_1, individual_2)
            u = np.random.uniform()
            
            # mutation in nature
            if u > mutate_percentage:
                new_individual = self.mutate(new_individual)
            
            self.add_new_population(new_individual)
            
        # logging
        score, best_individual = heapq.nlargest(1, self.population)[0]
        xs = self.country.cities[best_individual + [best_individual[0]], 0]
        ys = self.country.cities[best_individual + [best_individual[0]], 1]
        
        self.path_ax.clear()
        self.path_ax.scatter(xs, ys)
        self.path_ax.plot(xs, ys)
        
        self.scores.append(abs(score))
        self.score_ax.clear()
        self.score_ax.plot(self.scores)
        if self.debug:
            self.score_ax.hlines(self.ideal_distance, xmin=0, xmax=t + 1)
        if self.debug:
            print("Iteration {}: current path {}, shortest path {}".format(
                t + 1, 
                abs(score), 
                self.ideal_distance
            ))
        else:
            print("Iteration {}: current path {}".format(t + 1, abs(score)))
    
    def learn(self, 
              total_timestep=100,
              individual_per_epochs=10, 
              n_breed_per_epoch=10, 
              mutate_percentage=0.1,
              mode="show",
              filename="demo.gif"):
        ani = animation.FuncAnimation(
            self.fig, 
            self.generate_single_frame,
            total_timestep, 
            fargs=(individual_per_epochs, n_breed_per_epoch, mutate_percentage),
            interval=10
        )
        if mode == "save":
            ani.save(filename, fps=60)
        elif mode == "show":
            plt.show()