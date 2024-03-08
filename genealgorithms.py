import numpy as np
from  neuronetgene import NeuronetGene


class GeneAlgorithms:
    # layes is array of count nodes in the layer
    # layers include input layer
    @staticmethod
    def genetareRandomGene(layers):
        biases = [np.random.rand(y) for y in layers[1:]]

        for i in range(len(biases)):
            for j in range(len(biases[i])):
                biases[i][j] = biases[i][j] * 2 - 1

        weights = [np.random.rand(x, y) for x, y in zip(layers[:-1], layers[1:])]

        for i in range(len(weights)):
            for j in range(len(weights[i])):
                for k in range(len(weights[i][j])):
                    weights[i][j][k] = weights[i][j][k] * 2 - 1

        return NeuronetGene(biases, weights)
    
    @staticmethod
    def isAllGeneComplete(genes):
        for gene in genes:
            if gene.completed < 1:
                return False
        
        return True
    
    # сортировка 
    # вначале по убыванию completed
    # после по возврастанию fitness
    @staticmethod
    def sortGene(genes):
        # TODO: Test
        return sorted(genes, key=lambda x: (-x.completed, x.fitness if x.fitness is not None else float('inf')))