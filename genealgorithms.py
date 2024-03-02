import numpy as np
from  neuronetgene import NeuronetGene


class GeneAlgorithms:
    # layes is array of count nodes in the layer
    # layers include input layer
    @staticmethod
    def genetareRandomGene(layers):
        biases = [np.random.rand(y) for y in layers[1:]]
        weights = [np.random.rand(x, y) for x, y in zip(layers[:-1], layers[1:])]
        return NeuronetGene(biases, weights)
    
    @staticmethod
    def isAllGeneComplete(genes):
        for gene in genes:
            if gene.completed < 1:
                return False
        
        return True
    
    @staticmethod
    def mutate(gene : NeuronetGene):
        pass
    @staticmethod
    def crossover(gene1 : NeuronetGene, gene2: NeuronetGene):
        pass