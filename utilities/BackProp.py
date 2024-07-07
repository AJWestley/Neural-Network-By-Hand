from typing import Callable
import numpy as np
from utilities.FeedForward import feed_forward_with_tracking

def back(
    X: np.ndarray, 
    Y: np.ndarray, 
    weights: list, 
    biases: list, 
    learning_rate: float, 
    hidden_activation: Callable,
    output_activation: Callable,
    output_weight_derivative: Callable,
    output_bias_derivative: Callable,
    hidden_weight_derivative: Callable,
    hidden_bias_derivative: Callable,
    activation_derivative: Callable,
    cost = None
    ) -> None:
    ''' Performs one step of back propagation '''
    
    # Make predictions
    layer_outputs = feed_forward_with_tracking(X, weights, biases, hidden_activation, output_activation)
    
    # Gradient ascent on output layer
    dJ_dW = output_weight_derivative(layer_outputs[-2], layer_outputs[-1], Y)
    dJ_db = output_bias_derivative(layer_outputs[-1], Y)
    weights[-1] += learning_rate * clipped(dJ_dW)
    biases[-1] += learning_rate * clipped(dJ_db)
    
    n = len(weights) - 2
    
    # Gradient ascent on hidden layers
    for i in range(n, -1, -1):
        # print(i)
        # dJ_dW = hidden_weight_derivative(layer_outputs[i], Y, layer_outputs[-1], weights[i+1], layer_outputs[i+1], activation_derivative)
        dJ_db = hidden_bias_derivative(Y, layer_outputs[-1], weights[i+1], layer_outputs[i+1], activation_derivative)
        # weights[i] += learning_rate * clipped(dJ_dW)
        biases[i] += learning_rate * clipped(dJ_db)
    
    if cost:
        print(cost(Y, layer_outputs[-1]))

def clipped(gradients: np.ndarray, min_threshold: float = 1e-3, max_threshold: float = 5) -> np.ndarray:
    return np.clip(gradients, min_threshold, max_threshold)

