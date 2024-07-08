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
    activation_derivative: Callable
    ) -> None:
    ''' Performs one step of back propagation '''
    
    # Make predictions
    layer_outputs = feed_forward_with_tracking(X, weights, biases, hidden_activation, output_activation)
    
    # Gradient Ascent
    delta = Y - layer_outputs[-1]
    n = len(weights) - 1
    
    for i in range(n, -1, -1):
        weights[i] += learning_rate * __clipped(__weight_derivative(layer_outputs[i], delta))
        biases[i] += learning_rate * __bias_derivative(delta)
        delta = __delta_update(layer_outputs[i], delta, weights[i], activation_derivative)

def __clipped(gradients: np.ndarray, min_threshold: float = 1e-3, max_threshold: float = 5) -> np.ndarray:
    return np.clip(gradients, min_threshold, max_threshold)

def __weight_derivative(
    X: np.ndarray, 
    delta: np.ndarray
    ) -> np.ndarray:
    ''' The derivative of this function with respect to a layer's weights '''
    return X.T.dot(delta)

def __bias_derivative(delta: np.ndarray) -> np.ndarray:
    ''' The derivative of this function with respect to a layer's biases '''
    return delta.sum(axis=0)

def __delta_update(
        X: np.ndarray, 
        delta: np.ndarray,
        weights: np.ndarray,
        activation_derivative: Callable
        ) -> np.ndarray:
        ''' The derivative of this function with respect to a layer's weights '''
        return (delta.dot(weights.T)) * activation_derivative(X)
