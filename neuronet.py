from typing import Any
import numpy as np
from  neuromath import NeuroMath
from  neuronetgene import NeuronetGene
from genealgorithms import GeneAlgorithms

# нейросеть в общем
class GeenNeuronet:
    def __init__(self, gene : NeuronetGene):
        self.gene = gene
    
    # проход по нейронной сети 
    def run(self, inputs):
        # актуальные выходные данные на этом шагу
        passingValues = NeuroMath.sigmoidArray(inputs)
        
        for index in range(len(self.gene.biases)):
            passingValues = NeuroMath.sigmoidArray(
                np.dot(self.gene.weights[index].T, passingValues) + self.gene.biases[index]
            )
        
        print(passingValues)
        
        return passingValues
    
gene = GeneAlgorithms.genetareRandomGene([1, 3, 2, 1])
neu = GeenNeuronet(gene)

gene.print()

GeenNeuronet(neu.run([3]))