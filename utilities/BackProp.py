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
    weight_derivative: Callable,
    bias_derivative: Callable,
    activation_derivative: Callable
    ) -> None:
    ''' Performs one step of back propagation '''
    
    # Make predictions
    layer_outputs = feed_forward_with_tracking(X, weights, biases, hidden_activation, output_activation)
    
    # Gradient Ascent
    delta = Y - layer_outputs[-1]
    n = len(weights) - 1
    
    for i in range(n, -1, -1):
        weights[i] += learning_rate * clipped(weight_derivative(layer_outputs[i], delta))
        biases[i] += learning_rate * bias_derivative(delta)
        delta = (delta.dot(weights[i].T)) * activation_derivative(layer_outputs[i])

def clipped(gradients: np.ndarray, min_threshold: float = 1e-3, max_threshold: float = 5) -> np.ndarray:
    return np.clip(gradients, min_threshold, max_threshold)

