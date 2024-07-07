from typing import Callable
import numpy as np


def cost(labels: np.ndarray, predictions: np.ndarray) -> float:
    ''' The categorical cross entropy cost function '''
    return (labels * np.log(predictions)).sum()

def output_layer_weight_derivative(inputs_from_hidden: np.ndarray, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    ''' The derivative of this function with respect to the output layer weights '''
    z = labels - predictions
    return inputs_from_hidden.T.dot(z)

def output_layer_bias_derivative(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    ''' The derivative of this function with respect to the output layer biases '''
    return (labels - predictions).sum(axis=0)

def hidden_layer_weight_derivative(
    X: np.ndarray, 
    labels: np.ndarray, 
    predictions: np.ndarray, 
    output_weights: np.ndarray,
    output_values: np.ndarray, 
    activation_derivative: Callable
    ) -> np.ndarray:
    ''' The derivative of this function with respect to a hidden layer's weights '''
    l_min_pred = (labels - predictions)
    print(l_min_pred.shape)
    print(output_weights.T.shape)
    dotty = l_min_pred.dot(output_weights.T)
    hada = dotty * activation_derivative(output_values)
    return X.T.dot(hada)

def hidden_layer_bias_derivative(
    labels: np.ndarray, 
    predictions: np.ndarray, 
    output_weights: np.ndarray,
    output_values: np.ndarray, 
    activation_derivative: Callable
    ) -> np.ndarray:
    ''' The derivative of this function with respect to a hidden layer's biases '''
    return ((labels - predictions).dot(output_weights.T) * activation_derivative(output_values)).sum(axis=0)
