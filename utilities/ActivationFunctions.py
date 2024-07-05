from typing import Callable
import numpy as np

def hidden(chosen_function: str) -> Callable:
    ''' Provides the chosen activation function
        Options:
        - 'sigmoid': Logistic Sigmoid
        - 'relu': Rectified Linear Unit
        - 'tanh': Hyperbolic Tan
        - 'identity': Passes through the input without applying an activation function
    '''
    match chosen_function:
        case 'sigmoid':
            activation_func = __sigmoid
        case 'tanh':
            activation_func = __tanh
        case 'identity':
            activation_func = __identity
        case 'relu':
            activation_func = __relu
        case _:
            raise ValueError(f'Invalid activation function provided: {chosen_function}.')
    return activation_func

def output(chosen_function: str) -> Callable:
    ''' Provides the chosen output activation function
        Options:
        - 'sigmoid': Logistic Sigmoid (Reccommended for binary or multilabel classification)
        - 'softmax': SoftMax Function (Reccommended for multiclass classification)
        - 'identity': Passes through the input without applying an activation function (Reccommended for regression)
    '''
    match chosen_function:
        case 'sigmoid':
            activation_func = __sigmoid
        case 'softmax':
            activation_func = __softmax
        case 'identity':
            activation_func = __identity
        case _:
            raise ValueError(f'Invalid activation function provided: {chosen_function}.')
    return activation_func

def __softmax(x: np.ndarray) -> np.ndarray:
    ''' Softmax function of a given vector '''
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)

def __sigmoid(x: np.ndarray) -> np.ndarray:
    ''' Logistic sigmoid function of a given vector '''
    return 1 / (1 + np.exp(-x))

def __relu(x: np.ndarray) -> np.ndarray:
    ''' Rectified linear unit function of a given vector '''
    return x * (x > 0)

def __identity(x: np.ndarray) -> np.ndarray:
    ''' Returns the input vector unchanged '''
    return x

def __tanh(x: np.ndarray) -> np.ndarray:
    ''' Tanh of a given vector '''
    return np.tanh(x)