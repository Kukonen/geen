import numpy as np
from genealgorithms import GeneAlgorithms
from neuronet import GeenNeuronet
from neuronetgene import NeuronetGene
from neuromath import NeuroMath
from mutator import Mutator
from crossover import Crossover
import math

class GeenLearn:
    # distance is the number of test data that the gene passes
    def __init__(self, layers, genes, learnX, testX, learnY, testY, distance = 0.2, mutateChance = 0.05):
        self.genes = genes
        self.learnX = learnX
        self.testX = testX
        self.learnY = learnY
        self.testY = testY
        self.distance = distance
        self.mutateChance = mutateChance
        self.layers = layers
        
        self.meanFitness = 0
    
    # all process learning
    def learn(self):
        while GeneAlgorithms.isAllGeneComplete(self.genes):
            # steping from 0 to 100% in learning data
            for start in range(0, 100, int(self.distance * 100)):
                self.step(start / 100)
                # TODO: optimize
                self.genes = GeneAlgorithms.sortGene(self.genes)
                
            self.meanFitness = self.meanFitness / len(self.genes)
            self.select()
            self.meanFitness = 0
                
    # start equal completed distance
    # only gene with start == completed run in step
    def step(self, start):
        # TODO: optimise this!
        for gene in self.genes:
            if (gene.completed == start):
                gene = self.running(gene)
                # that's donesn't goes through 100%
                ranDistace = start + self.distance * 100 if start + self.distance * 100 < 100 else 100
                gene.completed += ranDistace
    
    # selection method
    def select(self):
        selection = []
        
        # TODO: maybe not all elements will be passed
        # select genes that's went to end
        for gene in self.genes:
            if gene.completed == 1:
                selection.append(gene)
                self.genes.remove(gene)
        
        # мы с некоторым шансом добавляем ген в выборку для селекции
        # зависимость от % пройденного пути и средней приспособленности
        for gene in self.genes:
            if NeuroMath.getRandomBooleanChoise(0.75 * gene.completed) and NeuroMath.getRandomBooleanChoise(-0.8 * (np.fabs(self.meanFitness - gene.fitness) - 1.05)):
                selection.append(gene)
                self.genes.remove(gene)
        
        # untill selection doesn't empty
        # doing crossover with all ended genes
        while len(selection):
            fisrtGeneIndex = 0
            secondGeneIndex = np.random.randint(1, len(selection))
            
            newGene1, newGene2 = Crossover.two_point_cross(selection[fisrtGeneIndex], selection[secondGeneIndex], self.layers)
            
            selection.pop(fisrtGeneIndex)
            selection.pop(secondGeneIndex)
            
            # mutate 
            if (NeuroMath.getRandomBooleanChoise(self.mutateChance)):
                newGene1 = Mutator.mutate(newGene1)
                
            if (NeuroMath.getRandomBooleanChoise(self.mutateChance)):    
                newGene2 = Mutator.mutate(newGene2)
            
            self.genes.append(newGene1)
            self.genes.append(newGene2)
        
    # passing distance and adding mistake to fitness        
    def running(self, gene: NeuronetGene):
        # TODO: now we set 0 after each step, maybe we need set all fitness with before steps
        gene.fitness = 0
        
        for index in range(
            int(len(self.learnX) * gene.completed), int(len(self.learnX) * (gene.completed + self.distance))
        ):
            gene.fitness += NeuroMath.MSE(
                GeenNeuronet.run(gene, self.learnX[index]), self.learnY[index]
            )
        
        # mean fitness
        gene.fitness = gene.fitness / (int(len(self.learnX) * (gene.completed + self.distance) - len(self.learnX) * gene.completed))
        
        # add gene mean fitness to all mean fitness
        self.meanFitness += gene.fitness
        return gene
    
    def getGenes(self):
        return self.genes