from typing import Callable
import numpy as np


def cost(labels: np.ndarray, predictions: np.ndarray) -> float:
    ''' The categorical cross entropy cost function '''
    return (labels * np.log(predictions)).sum()

def weight_derivative(
    X: np.ndarray, 
    delta: np.ndarray
    ) -> np.ndarray:
    ''' The derivative of this function with respect to a layer's weights '''
    return X.T.dot(delta)

def bias_derivative(delta: np.ndarray) -> np.ndarray:
    ''' The derivative of this function with respect to a layer's biases '''
    return delta.sum(axis=0)
