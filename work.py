# пример запуска нейросети

from neurolearn import GeenLearn
from genealgorithms import GeneAlgorithms

layers = [1, 3, 4, 2]
genes = []

for i in range(1000):
    genes.append(GeneAlgorithms.genetareRandomGene(layers))


learn = GeenLearn(layers, genes, learnX, testX, learnY, testY)
learn.learn()
print(learn.getTheBestGene().fitness)
