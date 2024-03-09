import random
from neuronetgene import NeuronetGene
import numpy as np


class Crossover:
    ##Я реализовал двухточечное скрещивание
    # @staticmethod
    # def two_point_cross(individual1: NeuronetGene, individual2: NeuronetGene, layers, with_elitisim = False):
    #     if with_elitisim:
    #         pass
    #     else:
    #         return Crossover.two_point_cross_two_genes(individual1, individual2, layers)

    # If var with_elitisim is True method will change only one gene that is not elit, in other case
    # method will change both genes
    # For stupid: first parameter is potentially elite, second is never
    @staticmethod
    def two_point_cross(individual1: NeuronetGene, individual2: NeuronetGene, layers, with_elitisim = False):
        weights_f = np.array(
            [item for sublist in [subsublist for sublist in individual1.weights for subsublist in sublist] for item in
             sublist])
        weights_s = np.array(
            [item for sublist in [subsublist for sublist in individual2.weights for subsublist in sublist] for item in
             sublist])
        biases_f = np.array([item for sublist in individual1.biases for item in sublist])
        biases_s = np.array([item for sublist in individual2.biases for item in sublist])

        if with_elitisim:
            weights_s = Crossover.transform_elitism(weights_f, weights_s)
            biases_s = Crossover.transform_elitism(biases_f, biases_s)
        else:
            weights_f, weights_s = Crossover.transform(weights_f, weights_s)
            biases_f, biases_s = Crossover.transform(biases_f, biases_s)

        counter = 0
        final_baises_f, final_baises_s = [], []

        for i in range(1, len(layers)):
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

    @staticmethod
    def transform_elitism(elit_arr, arr):
        length = len(elit_arr)

        bound1 = random.randint(0, length)
        bound2 = random.randint(0, length)

        if bound1 < bound2:
            bound1, bound2 = bound2, bound1

        for i in range(bound1, bound2):
            arr[i] = elit_arr[i]

        return arr