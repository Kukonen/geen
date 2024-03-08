import numpy as np
from genealgorithms import GeneAlgorithms
from neuronet import GeenNeuronet
from neuronetgene import NeuronetGene
from neuromath import NeuroMath
from mutator import Mutator
from crossover import Crossover
import matplotlib.pyplot as plt

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
        
        self.meanFitneses = []
    
    # all process learning
    # steps is the number of max steps
    def learn(self, steps = 1000):
        plt.figure()
        axis_x = []
        counter = 1
        
        # while not(GeneAlgorithms.isAllGeneComplete(self.genes)) and steps:
        while steps:
            print("steps have " + str(steps))
            # steping from 0 to 100% in learning data
            for start in range(0, 100, int(self.distance * 100)):
                self.step(start / 100)
                # TODO: optimize
                self.genes = GeneAlgorithms.sortGene(self.genes)
                
            self.meanFitneses.append(sum(gen.fitness for gen in self.genes) / len(self.genes))
            axis_x.append(counter)
            
            self.select()
            
            plt.clf()
            plt.plot(axis_x, self.meanFitneses, marker='o')  # Строим график с точками

            plt.title("График приспосабливаемости")
            plt.xlabel('Поколения')
            plt.ylabel('Общая приспосабливаемость')

            plt.pause(0.5)

            steps -= 1
            counter += 1
        
        plt.show()
                
    # start equal completed distance
    # only gene with start == completed run in step
    def step(self, start):
        # TODO: optimise this!
        for index, gene in enumerate(self.genes):
            if (gene.completed == start):
                self.genes[index] = self.running(gene)
                # that's donesn't goes through 100%
                ranDistace = start + self.distance * 100 if start + self.distance * 100 < 100 else 100
                self.genes[index].completed += ranDistace
    
    # selection method
    def select(self):
        selection = []
        
        # TODO: maybe not all elements will be passed
        # select genes that's went to end
        # for gene in self.genes:
        #     if gene.completed == 1:
        #         selection.append(gene)
        #         self.genes.remove(gene)
        
        # мы с некоторым шансом добавляем ген в выборку для селекции
        # зависимость от % пройденного пути и средней приспособленности
        for gene in self.genes:
            # if NeuroMath.getRandomBooleanChoise(0.75 * gene.completed) and NeuroMath.getRandomBooleanChoise(-0.8 * (gene.fitness - 1.05)):
            if NeuroMath.getRandomBooleanChoise(0.85 * gene.completed) and NeuroMath.getRandomBooleanChoise(-0.9 * (gene.fitness - 1.05)):
                selection.append(gene)
                self.genes.remove(gene)
        
        # untill selection doesn't empty
        # doing crossover with all ended genes
        print("selection start with " + str(len(selection)))
        while len(selection) > 1:
            
            fisrtGeneIndex = 0
            secondGeneIndex = np.random.randint(1, len(selection))
            
            newGene1, newGene2 = Crossover.two_point_cross(selection[fisrtGeneIndex], selection[secondGeneIndex], self.layers)
            
            
            # mutate 
            # if (NeuroMath.getRandomBooleanChoise(self.mutateChance)):
            #     newGene1 = Mutator.mutate(newGene1)
                
            # if (NeuroMath.getRandomBooleanChoise(self.mutateChance)):    
            #     newGene2 = Mutator.mutate(newGene2)
                
            newGene1 = Mutator.mutate(newGene1)
            newGene2 = Mutator.mutate(newGene2)
            
            self.genes.append(newGene1)
            self.genes.append(newGene2)
            
            selection.pop(secondGeneIndex)
            selection.pop(fisrtGeneIndex)
        if len(selection):
            self.genes.append(selection[0])
        
        print("selection end")
        
    # TODO: here learn is number and it's nesting in array, but output can have many values
    # passing distance and adding mistake to fitness        
    def running(self, gene: NeuronetGene):
        # TODO: now we set 0 after each step, maybe we need set all fitness with before steps
        gene.fitness = 0
        
        for index in range(
            int(len(self.learnX) * gene.completed), int(len(self.learnX) * (gene.completed + self.distance if gene.completed + self.distance < 1 else 1))
        ):
            gene.fitness += NeuroMath.MSE(
                GeenNeuronet.run(gene, self.learnX[index]), np.array(self.learnY[index])
            )
        
        # mean fitness
        gene.fitness = gene.fitness / (int(len(self.learnX) * (gene.completed + self.distance if gene.completed + self.distance < 1 else 1) - len(self.learnX) * gene.completed))
        
        return gene
    
    def getGenes(self):
        return self.genes
    
    # TODO: it's just select with the best fitness
    def getTheBestGene(self):
        for index, gene in enumerate(self.genes):
            # self.genes[index].completed = 0
            # self.genes[index] = self.running(self.genes[index])
            gene.completed = 0
            self.genes[index] = self.running(gene)

        # return self.genes
        # return self.genes[0]
        return sorted(self.genes, key=lambda x: -(x.fitness if x.fitness is not None else float('-inf')))