import numpy as np
from genealgorithms import GeneAlgorithms
from neuronet import GeenNeuronet
from neuronetgene import NeuronetGene
from neuromath import NeuroMath
class GeenLearn:
    # distance is the number of test data that the gene passes
    def __init__(self, genes, learnX, testX, learnY, testY, distance = 0.2):
        self.genes = genes
        self.learnX = learnX
        self.testX = testX
        self.learnY = learnY
        self.testY = testY
        self.distance = distance
    
    # all process learning
    # survival is % best what will be survived
    def learn(self, survive):
        while GeneAlgorithms.isAllGeneComplete(self.genes):
            # steping from 0 to 100% in learning data
            for i in range(1):
                pass
    
    # start equal completed distance
    # only gene with start == completed run in step
    def step(self, start):
        for gene in self.genes:
            if (gene.completed == start):
                gene = self.running(gene)
        
    # passing distance and adding mistake to fitness        
    def running(self, gene: NeuronetGene):
        for index in range(
            int(len(self.learnX) * gene.completed), int(len(self.learnX) * (gene.completed + self.step))
        ):
            gene.fitness += NeuroMath.MSE(
                GeenNeuronet.run(gene, self.learnX[index]), self.learnY[index]
            )
        
        return gene