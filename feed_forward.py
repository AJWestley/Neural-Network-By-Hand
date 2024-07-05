from typing import Callable
import numpy as np
import neural_net_utils

def feed_forward(
    X: np.ndarray, 
    W: tuple[np.ndarray], 
    b: tuple[np.ndarray], 
    hidden_activation: str = 'tanh', 
    output_activation: str = 'softmax') -> np.ndarray:
    ''' Performs the feed forward prediction algorithm in its entirety '''
    
    __check_weight_formats(W, b)
    
    hidden_layer_act = neural_net_utils.hidden_activation(hidden_activation)
    output_layer_act = neural_net_utils.output_activation(output_activation)
    
    current = X
    for i in range(len(W)-1):
        current = __forward(current, W[i], b[i], hidden_layer_act)
    return __predict(current, W[-1], b[-1], output_layer_act)

def __forward(
    X: np.ndarray, 
    W: np.ndarray, 
    b: np.ndarray, 
    activation_function: Callable
    ) -> np.ndarray:
    ''' Performs one step of feed forward for the hidden layers '''
    return activation_function(X.dot(W) + b)

def __predict(X: np.ndarray, V: np.ndarray, b: np.ndarray, activation_function: Callable) -> np.ndarray:
    ''' Performs the final step of feed forward for the output layer '''
    return activation_function(X.dot(V) + b)

def __check_weight_formats(W: tuple, b: tuple) -> None:
    ''' Error checking for the weight and bias tuples '''
    if len(W) != len(b):
        raise ValueError('Ws and bs are not the same length.')
    
    if len(W) != len(b):
        raise ValueError('Ws and bs are not the same length.')