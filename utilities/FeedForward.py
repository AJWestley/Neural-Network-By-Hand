from typing import Callable
import numpy as np

def feed_forward(
    X: np.ndarray, 
    W: list, 
    b: list, 
    hidden_activation: Callable, 
    output_activation: Callable
    ) -> np.ndarray:
    ''' Performs the feed forward prediction algorithm in its entirety '''
    
    __check_weight_formats(W, b)
    
    # Propagate the signal through the network
    current = X
    for i in range(len(W)-1):
        current = __forward(current, W[i], b[i], hidden_activation)
    return __predict(current, W[-1], b[-1], output_activation)

def feed_forward_with_tracking(
    X: np.ndarray, 
    W: list, 
    b: list, 
    hidden_activation: Callable, 
    output_activation: Callable
    ) -> list:
    ''' Performs the feed forward prediction algorithm in its entirety while tracking all layer outputs'''
    
    __check_weight_formats(W, b)
    
    layer_outputs = [X]
    for i in range(len(W)-1):
        layer_outputs.append(__forward(layer_outputs[-1], W[i], b[i], hidden_activation))
    layer_outputs.append(__predict(layer_outputs[-1], W[-1], b[-1], output_activation))
    return layer_outputs

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

def __check_weight_formats(W: list, b: list) -> None:
    ''' Error checking for the weight and bias lists '''
    if len(W) != len(b):
        raise ValueError('Ws and bs are not the same length.')
    
    if len(W) != len(b):
        raise ValueError('Ws and bs are not the same length.')