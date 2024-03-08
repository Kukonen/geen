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
    
    @staticmethod
    def getRandomBooleanChoise(chance):
        if chance >= 1:
            return True
        
        if chance < 0:
            return False
        
        choices = [True, False]
        return np.random.choice(choices, p=[chance, 1 - chance])