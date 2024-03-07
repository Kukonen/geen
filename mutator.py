import random
from geen.neuronetgene import NeuronetGene


class Mutator:
    @staticmethod
    def mutate(individual: NeuronetGene, mean=0, std_deviation=1, mutation_rate=0.1):
        mutated_biases = Mutator.mutateBiases(individual.biases, mean, std_deviation, mutation_rate)
        mutated_weights = Mutator.mutateWeights(individual.weights, mean, std_deviation, mutation_rate)

        return NeuronetGene(mutated_biases, mutated_weights)

    @staticmethod
    def mutateBiases(biases, mean, std_deviation, mutation_rate):
        mutated_biases = []

        for layer in biases:
            mutated_layer = []
            for biase in layer:
                if random.random() < mutation_rate:
                    biase = biase + random.gauss(mean, std_deviation)
                    mutated_layer.append(biase)
                else:
                    mutated_layer.append(biase)

            mutated_biases.append(mutated_layer)

        return mutated_biases

    @staticmethod
    def mutateWeights(weights, mean, std_deviation, mutation_rate):
        mutated_weights = []

        for layer in weights:
            mutated_layer = []
            for neuron_link in layer:
                mutated_neuron_link = []
                for weight in neuron_link:
                    if random.random() < mutation_rate:
                        weight = weight + random.gauss(mean, std_deviation)
                        mutated_neuron_link.append(weight)
                    else:
                        mutated_neuron_link.append(weight)

                mutated_layer.append(mutated_neuron_link)

            mutated_weights.append(mutated_layer)

        return mutated_weights
