# пример запуска нейросети
from geen.datasources import DataSet
from neurolearn import GeenLearn
from genealgorithms import GeneAlgorithms

layers = [1, 3, 4, 2]
genes = []

for i in range(1000):
    genes.append(GeneAlgorithms.genetareRandomGene(layers))

learn_data, test_data = DataSet.get_data("student-mat.xlsx")

learn = GeenLearn(layers, genes, learn_data[0], test_data[0], learn_data[1], test_data[1])
learn.learn()
print(learn.getGenes()[0].fitness)
