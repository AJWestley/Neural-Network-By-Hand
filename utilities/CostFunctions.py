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
                return BinaryCrossEntropy.cost
            elif output_activation == 'identity':
                return MeanSquaredError.cost
            raise ValueError(f'Invalid activation function provided: {chosen_function}.')
        case 'bce':
            return BinaryCrossEntropy.cost
        case 'cce':
            return CategoricalCrossEntropy.cost
        case 'mse':
            return MeanSquaredError.cost
        case _:
            raise ValueError(f"The invalid cost function provided: {chosen_function}.")

def delta_update(chosen_function: str, output_activation: str) -> Callable:
    ''' Acts as a dictionary of cost function delta updates '''
    match chosen_function:
        case 'auto':
            if output_activation == 'softmax':
                return CategoricalCrossEntropy.delta_update
            elif output_activation == 'sigmoid':
                raise NotImplementedError()
            elif output_activation == 'identity':
                raise NotImplementedError()
            raise ValueError(f'Invalid activation function provided: {chosen_function}.')
        case 'bce':
            raise NotImplementedError()
        case 'cce':
            return CategoricalCrossEntropy.delta_update
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
    def delta_update(
        X: np.ndarray, 
        delta: np.ndarray,
        weights: np.ndarray,
        activation_derivative: Callable
        ) -> np.ndarray:
        ''' The derivative of this function with respect to a layer's weights '''
        return (delta.dot(weights.T)) * activation_derivative(X)

class BinaryCrossEntropy:
    ''' Utilities for Binary Cross Entropy cost function '''
    
    @staticmethod
    def cost(labels: np.ndarray, predictions: np.ndarray) -> float:
        ''' The binary cross entropy cost function '''
        return (labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions)).sum()

    @staticmethod
    def delta_update(
        X: np.ndarray, 
        delta: np.ndarray,
        weights: np.ndarray,
        activation_derivative: Callable
        ) -> np.ndarray:
        ''' The derivative of this function with respect to a layer's weights '''
        return (delta.dot(weights.T)) * activation_derivative(X)

class MeanSquaredError:
    ''' Utilities for Mean Squared Error cost function '''
    
    @staticmethod
    def cost(labels: np.ndarray, predictions: np.ndarray) -> float:
        ''' The mean squared error cost function '''
        n, _ = labels.shape
        return - np.square(labels - predictions).sum() / n

    @staticmethod
    def delta_update(
        X: np.ndarray, 
        delta: np.ndarray,
        weights: np.ndarray,
        activation_derivative: Callable
        ) -> np.ndarray:
        ''' The derivative of this function with respect to a layer's weights '''
        return (delta.dot(weights.T)) * activation_derivative(X)