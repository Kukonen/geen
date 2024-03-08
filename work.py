# пример запуска нейросети
from datasources import DataSet
from neurolearn import GeenLearn
from genealgorithms import GeneAlgorithms
from neuronet import GeenNeuronet

layers = [19, 20, 15, 10, 5, 1]
genes = []

for i in range(150):
    genes.append(GeneAlgorithms.genetareRandomGene(layers))

learn_data, test_data = DataSet.get_data("student-mat.xlsx")

learn = GeenLearn(layers, genes, learn_data[0], test_data[0], learn_data[1], test_data[1])
learn.learn(10)
print(len(learn.genes))
genes = learn.getTheBestGene()
bestGene = genes[0]
worstGene = genes[-1]
print(bestGene.fitness, worstGene.fitness)
for i in range(int(len(test_data[1]) * 0.2)):
    print("best: ", GeenNeuronet.run(bestGene, test_data[0][i]))
    print("worst: ",GeenNeuronet.run(worstGene, test_data[0][i]))
    print("test: ",test_data[1][i])
