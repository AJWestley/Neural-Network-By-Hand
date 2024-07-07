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
    layer_outputs = feed_forward_with_tracking(X, weights, biases, hidden_activation, output_activation)
    weights[-1] += learning_rate * output_weight_derivative(layer_outputs[-2], layer_outputs[-1], Y)
    biases[-1] += learning_rate * output_bias_derivative(layer_outputs[-1], Y)
    # for i in range(0, -1, -1):
    for i in range(1, 2, -1):
        weights[i] += learning_rate * hidden_weight_derivative(layer_outputs[i], Y, layer_outputs[-1], weights[i+1], layer_outputs[i+1], activation_derivative)
        biases[i] += learning_rate * hidden_bias_derivative(Y, layer_outputs[-1], weights[i+1], layer_outputs[i+1], activation_derivative)
    
    if cost:
        print(cost(Y, layer_outputs[-1]))

