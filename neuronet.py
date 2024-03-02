from typing import Any
import numpy as np
from  neuromath import NeuroMath
from  neuronetgene import NeuronetGene
from genealgorithms import GeneAlgorithms

# нейросеть в общем
class GeenNeuronet:
    
    # проход по нейронной сети 
    @staticmethod
    def run(gene: NeuronetGene, inputs):
        # актуальные выходные данные на этом шагу
        passingValues = NeuroMath.sigmoidArray(inputs)
        
        for index in range(len(gene.biases)):
            passingValues = NeuroMath.sigmoidArray(
                np.dot(gene.weights[index].T, passingValues) + gene.biases[index]
            )
        
        print(passingValues)
        
        return passingValues
    
gene = GeneAlgorithms.genetareRandomGene([1, 3, 2, 1])
neu = GeenNeuronet(gene)

gene.print()

GeenNeuronet(neu.run([3]))