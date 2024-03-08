# пример запуска нейросети
from datasources import DataSet
from neurolearn import GeenLearn
from genealgorithms import GeneAlgorithms
from neuronet import GeenNeuronet

layers = [10, 15, 10, 10, 5, 1]
genes = []

for i in range(150):
    genes.append(GeneAlgorithms.genetareRandomGene(layers))

# learn_data, test_data = DataSet.get_data("student-mat.xlsx")

# learn = GeenLearn(layers, genes, learn_data[0], test_data[0], learn_data[1], test_data[1])

X_train, X_test, y_train, y_test = DataSet.getRandomClassificationData()

learn = GeenLearn(layers, genes, X_train, X_test, y_train, y_test, distance=0.015)
learn.learn(500)
print(len(learn.genes))
genes = learn.getTheBestGene()
bestGene = genes[0]
worstGene = genes[-1]
print(bestGene.fitness, worstGene.fitness)
for i in range(int(len(X_test[1]) * 0.2)):
    print("best: ", GeenNeuronet.run(bestGene, X_test[i]))
    print("worst: ",GeenNeuronet.run(worstGene, X_test[i]))
    print("test: ",y_test[i])
