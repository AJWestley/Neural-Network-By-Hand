from typing import Callable
import numpy as np

#TODO: Implement the other cost functions

def cost_function(chosen_function: str, output_activation: str) -> Callable:
    ''' Acts as a dictionary of cost functions '''
    match chosen_function:
        case 'auto':
            if output_activation == 'softmax':
                return CategoricalCrossEntropy.cost
            elif output_activation == 'sigmoid':
                raise NotImplementedError()
            elif output_activation == 'identity':
                raise NotImplementedError()
            raise ValueError(f'Invalid activation function provided: {chosen_function}.')
        case 'bce':
            raise NotImplementedError()
        case 'cce':
            return CategoricalCrossEntropy.cost
        case 'mse':
            raise NotImplementedError()
        case _:
            raise ValueError(f"The invalid cost function provided: {chosen_function}.")

def weight_derivative(chosen_function: str, output_activation: str) -> Callable:
    ''' Acts as a dictionary of cost function weight derivatives '''
    match chosen_function:
        case 'auto':
            if output_activation == 'softmax':
                return CategoricalCrossEntropy.weight_derivative
            elif output_activation == 'sigmoid':
                raise NotImplementedError()
            elif output_activation == 'identity':
                raise NotImplementedError()
            raise ValueError(f'Invalid activation function provided: {chosen_function}.')
        case 'bce':
            raise NotImplementedError()
        case 'cce':
            return CategoricalCrossEntropy.weight_derivative
        case 'mse':
            raise NotImplementedError()
        case _:
            raise ValueError(f"The invalid cost function provided: {chosen_function}.")

def bias_derivative(chosen_function: str, output_activation: str) -> Callable:
    ''' Acts as a dictionary of cost functions '''
    match chosen_function:
        case 'auto':
            if output_activation == 'softmax':
                return CategoricalCrossEntropy.bias_derivative
            elif output_activation == 'sigmoid':
                raise NotImplementedError()
            elif output_activation == 'identity':
                raise NotImplementedError()
            raise ValueError(f'Invalid activation function provided: {chosen_function}.')
        case 'bce':
            raise NotImplementedError()
        case 'cce':
            return CategoricalCrossEntropy.bias_derivative
        case 'mse':
            raise NotImplementedError()
        case _:
            raise ValueError(f"The invalid cost function provided: {chosen_function}.")

class CategoricalCrossEntropy:
    ''' Utilities for Categorical Cross Entropy cost function '''
    
    @staticmethod
    def cost(labels: np.ndarray, predictions: np.ndarray) -> float:
        ''' The categorical cross entropy cost function '''
        return (labels * np.log(predictions)).sum()

    @staticmethod
    def weight_derivative(
        X: np.ndarray, 
        delta: np.ndarray
        ) -> np.ndarray:
        ''' The derivative of this function with respect to a layer's weights '''
        return X.T.dot(delta)

    @staticmethod
    def bias_derivative(delta: np.ndarray) -> np.ndarray:
        ''' The derivative of this function with respect to a layer's biases '''
        return delta.sum(axis=0)

class BinaryCrossEntropy:
    ''' Utilities for Binary Cross Entropy cost function '''
    
    @staticmethod
    def cost(labels: np.ndarray, predictions: np.ndarray) -> float:
        ''' The binary cross entropy cost function '''
        return (labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions)).sum()

    @staticmethod
    def weight_derivative(
        X: np.ndarray, 
        delta: np.ndarray
        ) -> np.ndarray:
        ''' The derivative of this function with respect to a layer's weights '''
        return X.T.dot(delta)

    @staticmethod
    def bias_derivative(delta: np.ndarray) -> np.ndarray:
        ''' The derivative of this function with respect to a layer's biases '''
        return delta.sum(axis=0)

class MeanSquaredError:
    ''' Utilities for Mean Squared Error cost function '''
    
    @staticmethod
    def cost(labels: np.ndarray, predictions: np.ndarray) -> float:
        ''' The mean squared error cost function '''
        n, _ = labels.shape
        return - np.square(labels - predictions).sum() / n

    @staticmethod
    def weight_derivative(
        X: np.ndarray, 
        delta: np.ndarray
        ) -> np.ndarray:
        ''' The derivative of this function with respect to a layer's weights '''
        return X.T.dot(delta)

    @staticmethod
    def bias_derivative(delta: np.ndarray) -> np.ndarray:
        ''' The derivative of this function with respect to a layer's biases '''
        return delta.sum(axis=0)