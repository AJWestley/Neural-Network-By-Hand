from typing import Callable
import numpy as np

def hidden_activation(chosen_function: str) -> Callable:
    ''' Provides the chosen activation function
        Options:
        - 'sigmoid': Logistic Sigmoid
        - 'relu': Rectified Linear Unit
        - 'tanh': Hyperbolic Tan
        - 'identity': Passes through the input without applying an activation function
    '''
    match chosen_function:
        case 'sigmoid':
            activation_func = lambda x: 1 / (1 + np.exp(-x))
        case 'tanh':
            activation_func = lambda x: np.tanh(x)
        case 'identity':
            activation_func = lambda x: x
        case 'relu':
            activation_func = lambda x: x * (x > 0)
        case _:
            raise ValueError(f'Invalid activation function provided: {chosen_function}.')
    return activation_func

def output_activation(chosen_function: str) -> Callable:
    ''' Provides the chosen output activation function
        Options:
        - 'sigmoid': Logistic Sigmoid (Reccommended for binary or multilabel classification)
        - 'softmax': SoftMax Function (Reccommended for multiclass classification)
        - 'identity': Passes through the input without applying an activation function (Reccommended for regression)
    '''
    match chosen_function:
        case 'sigmoid':
            activation_func = lambda x: 1 / (1 + np.exp(-x))
        case 'softmax':
            activation_func = __softmax
        case 'identity':
            activation_func = lambda x: x
        case _:
            raise ValueError(f'Invalid activation function provided: {chosen_function}.')
    return activation_func

def __softmax(x: np.ndarray) -> np.ndarray:
    ''' Softmax function of a given vector '''
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)