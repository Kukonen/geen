import numpy as np

class NeuronetGene:
    biases = np.array([])
    weights = np.array([])
    # % of completed data
    completed = 0
    # fitness by completed data procent
    fitness = None
    
    def __init__(self, biases, weights):
        self.biases = biases
        self.weights = weights
    
    def changeFitness(self, completed, fitness):
        self.completed = completed
        self.fitness = fitness
        
    def clearFitness(self):
        self.completed = 0
        self.fitness = None
        
    def print(self):
        print(self.biases)
        print(self.weights)
        print(self.completed)
        print(self.fitness)
        
# GeenHelper.genetareRandomGene([2, 4, 3, 2]).print()