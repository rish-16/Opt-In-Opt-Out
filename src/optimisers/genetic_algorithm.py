import math
import numpy as np

class GeneticAlgorithmSolver:
    def __init__(self, generations, popsize, elite_size):
        self.generations = generations
        self.popsize = popsize
        self.elite_size = elite_size

        assert self.popsize > self.elite_size, "Elite population must be smaller than original population."

    def run(self, agent_class, fit, mate, mut, scrit=None):
        '''
        create population -> fitness -> selection -> mate -> mutate
        '''
        population = [agent_class for _ in range(self.popsize)]
        for gen in range(self.generations):
            population_scores = fit(population) # [agent, score]
            sorted_pop = list(sorted(population_scores, reverse=True, key=lambda x : x[1])) # sort by score
            
            def get_agents(arr):
                agents = []
                for i in range(len(arr)):
                    agents.append(arr[i][0])

                return agents
            
            elite_pop = get_agents(sorted_pop)[:self.elite_size] # get top agents
            population = mate(population)
            population = mut(population)

        return population
            

        