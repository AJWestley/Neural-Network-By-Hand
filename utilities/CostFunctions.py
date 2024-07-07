import numpy as np

class CategoricalCrossEntropy:
    ''' Utilities for Categorical Cross Entropy cost function '''
    
    @staticmethod
    def cost(labels: np.ndarray, predictions: np.ndarray) -> float:
        ''' The categorical cross entropy cost function '''
        return (labels * np.log(predictions)).sum()

    @staticmethod
    def weight_derivative(
        X: np.ndarray, 
        delta: np.ndarray
        ) -> np.ndarray:
        ''' The derivative of this function with respect to a layer's weights '''
        return X.T.dot(delta)

    @staticmethod
    def bias_derivative(delta: np.ndarray) -> np.ndarray:
        ''' The derivative of this function with respect to a layer's biases '''
        return delta.sum(axis=0)
