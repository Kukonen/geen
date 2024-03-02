import numpy as np

class NeuroMath:
    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def sigmoidArray(arr):
        return [NeuroMath.sigmoid(x) for x in arr]
    @staticmethod
    def MSE(output, target):
        return ((target - output)**2).mean(axis=None)