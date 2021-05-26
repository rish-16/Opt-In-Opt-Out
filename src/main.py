from optimisers.simulated_annealing import SimulatedAnnealingSolver
import numpy as np

target = 40
def evaluate(val):
    return abs(val - target)

def mutate(val):
    return val - np.random.uniform(0, 1, [1])[0]

solver = SimulatedAnnealingSolver(1000, 1, 0.1)

for i in range(10):
    best = solver.run(80, evaluate, mutate, lambda v : abs(v - target) <= 1)
    print (best, target)