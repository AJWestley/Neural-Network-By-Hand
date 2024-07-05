from typing import Callable
import numpy as np

def generator(chosen_function: str, activation_function: str):
    match chosen_function:
        case 'auto':
            if activation_function in ['tanh', 'sigmoid']:
                return __norm_xavier
            else:
                return __he
        case 'he':
            return __he
        case 'xavier':
            return __norm_xavier
        case 'uniform':
            return __uniform
        case _:
            raise ValueError(f'Invalid weight initialisation function provided: {chosen_function}.')

def layer_weights(in_size: int, out_size: int, generator: Callable) -> np.ndarray:
    ''' Generates a weight matrix for one hidden layer '''
    return generator(-1, 1, (in_size, out_size))

def layer_biases(size: int) -> np.ndarray:
    ''' Generates a bias matrix for one hidden layer '''
    return np.zeros((size, 1))

def __uniform(in_size: int, out_size: int) -> np.ndarray:
    ''' Generates a weight matrix uniform on the interval [-1, 1] '''
    return np.random.uniform(-1, 1, (in_size, out_size))

def __norm_xavier(in_size: int, out_size: int) -> np.ndarray:
    ''' Generates a weight matrix according to normalised Xavier initialisation '''
    bound = np.sqrt(6) / np.sqrt(in_size + out_size)
    return np.random.uniform(-bound, bound, (in_size, out_size))

def __he(in_size: int, out_size: int) -> np.ndarray:
    ''' Generates a weight matrix according to He initialisation '''
    std = np.sqrt(in_size)
    return np.random.normal(0, std, (in_size, out_size))