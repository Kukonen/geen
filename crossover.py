import random
from geen.neuronetgene import NeuronetGene
import numpy as np


class Crossover:
    ##Я реализовал двухточечное скрещивание
    @staticmethod
    def two_point_cross(individual1: NeuronetGene, individual2: NeuronetGene, layers):
        weights_f = np.reshape(individual1.weights, -1)
        weights_s = np.reshape(individual2.weights, -1)
        biases_f = np.reshape(individual1.biases, -1)
        biases_s = np.reshape(individual2.biases, -1)

        weights_f, weights_s = Crossover.transform(weights_f, weights_s)
        biases_f, biases_s = Crossover.transform(biases_f, biases_s)

        counter = 0
        final_baises_f, final_baises_s = [], []

        for i in range(1, len(layers) - 1):
            temp_array_f, temp_array_s = [], []
            for j in range(0, layers[i]):
                temp_array_f.append(biases_f[counter])
                temp_array_s.append(biases_s[counter])
                counter += 1

            final_baises_f.append(temp_array_f)
            final_baises_s.append(temp_array_s)

        counter = 0
        final_weights_f, final_weights_s = [], []

        for i in range(0, len(layers) - 1):
            temp_array_f, temp_array_s = [], []
            for j in range(0, layers[i]):
                temp_subarray_f, temp_subarray_s = [], []
                for k in range(0, layers[i + 1]):
                    temp_subarray_f.append(weights_f[counter])
                    temp_subarray_s.append(weights_s[counter])
                    counter += 1

                temp_array_f.append(temp_subarray_f)
                temp_array_s.append(temp_subarray_s)

            final_weights_f.append(temp_array_f)
            final_weights_s.append(temp_array_s)

        return NeuronetGene(final_baises_f, final_weights_f), NeuronetGene(final_baises_s, final_weights_s)


    @staticmethod
    def transform(arr1, arr2):
        length = len(arr1)

        bound1 = random.randint(0, length)
        bound2 = random.randint(0, length)

        if bound1 < bound2:
            bound1, bound2 = bound2, bound1

        for i in range(bound1, bound2):
            arr1[i], arr2[i] = arr2[i], arr1[i]

        return arr1, arr2