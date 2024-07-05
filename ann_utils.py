from typing import Callable
import numpy as np
from ann_exceptions import FunctionNotAdvisedWarning

# ----- Cost Functions ----- #

class CostFunctions:
    
    @staticmethod
    def cost_function(chosen_function: str) -> Callable:
        raise NotImplementedError()

    @staticmethod
    def cost_derivative(chosen_function: str) -> Callable:
        raise NotImplementedError()

    # For Regression
    
    @staticmethod
    def __mean_squared_error(
        weights: np.ndarray, 
        biases: np.ndarray, 
        sample_input: np.ndarray, 
        expected_output: np.ndarray,
        regression: bool
        ) -> float:
        if not regression:
            raise FunctionNotAdvisedWarning('The mean squared error function is not advised for classification tasks')
        
        raise NotImplementedError()

    @staticmethod
    def __mean_squared_error_derivative(
        weights: np.ndarray, 
        biases: np.ndarray, 
        sample_input: np.ndarray, 
        expected_output: np.ndarray,
        regression: bool
        ) -> float:
        if not regression:
            raise FunctionNotAdvisedWarning('The mean squared error function is not advised for classification tasks')
        
        raise NotImplementedError()

    @staticmethod
    def __mean_absolute_error(
        weights: np.ndarray, 
        biases: np.ndarray, 
        sample_input: np.ndarray, 
        expected_output: np.ndarray,
        regression: bool
        ) -> float:
        if not regression:
            raise FunctionNotAdvisedWarning('The mean absolute error function is not advised for classification tasks')
        
        raise NotImplementedError()

    @staticmethod
    def __mean_absolute_error_derivative(
        weights: np.ndarray, 
        biases: np.ndarray, 
        sample_input: np.ndarray, 
        expected_output: np.ndarray,
        regression: bool
        ) -> float:
        if not regression:
            raise FunctionNotAdvisedWarning('The mean squared error function is not advised for classification tasks')
        
        raise NotImplementedError()

    @staticmethod
    def __huber_loss(
        weights: np.ndarray, 
        biases: np.ndarray, 
        sample_input: np.ndarray, 
        expected_output: np.ndarray,
        regression: bool
        ) -> float:
        if not regression:
            raise FunctionNotAdvisedWarning('The huber loss function is not advised for classification tasks')
        
        raise NotImplementedError()

    @staticmethod
    def __huber_loss_derivative(
        weights: np.ndarray, 
        biases: np.ndarray, 
        sample_input: np.ndarray, 
        expected_output: np.ndarray,
        regression: bool
        ) -> float:
        if not regression:
            raise FunctionNotAdvisedWarning('The huber loss function is not advised for classification tasks')
        
        raise NotImplementedError()

    # For Classification
    
    @staticmethod
    def __binary_cross_entropy(
        weights: np.ndarray, 
        biases: np.ndarray, 
        sample_input: np.ndarray, 
        expected_output: np.ndarray,
        regression: bool
        ) -> float:
        if regression:
            raise FunctionNotAdvisedWarning('The binary cross entropy function is not advised for regression tasks')
        
        raise NotImplementedError()

    @staticmethod
    def __binary_cross_entropy_derivative(
        weights: np.ndarray, 
        biases: np.ndarray, 
        sample_input: np.ndarray, 
        expected_output: np.ndarray,
        regression: bool
        ) -> float:
        if regression:
            raise FunctionNotAdvisedWarning('The binary cross entropy function is not advised for regression tasks')
        
        raise NotImplementedError()

    @staticmethod
    def __categorical_cross_entropy(
        weights: np.ndarray, 
        biases: np.ndarray, 
        sample_input: np.ndarray, 
        expected_output: np.ndarray,
        regression: bool
        ) -> float:
        if regression:
            raise FunctionNotAdvisedWarning('The categorical cross entropy function is not advised for regression tasks')
        
        raise NotImplementedError()

    @staticmethod
    def __categorical_cross_entropy_derivative(
        weights: np.ndarray, 
        biases: np.ndarray, 
        sample_input: np.ndarray, 
        expected_output: np.ndarray,
        regression: bool
        ) -> float:
        if regression:
            raise FunctionNotAdvisedWarning('The categorical cross entropy function is not advised for regression tasks')
        
        raise NotImplementedError()

    @staticmethod
    def __hinge_loss(
        weights: np.ndarray, 
        biases: np.ndarray, 
        sample_input: np.ndarray, 
        expected_output: np.ndarray,
        regression: bool
        ) -> float:
        if regression:
            raise FunctionNotAdvisedWarning('The hinge loss function is not advised for regression tasks')
        
        raise NotImplementedError()

    @staticmethod
    def __hinge_loss_derivative(
        weights: np.ndarray, 
        biases: np.ndarray, 
        sample_input: np.ndarray, 
        expected_output: np.ndarray,
        regression: bool
        ) -> float:
        if regression:
            raise FunctionNotAdvisedWarning('The chinge loss function is not advised for regression tasks')
        
        raise NotImplementedError()

    @staticmethod
    def __log_loss(
        weights: np.ndarray, 
        biases: np.ndarray, 
        sample_input: np.ndarray, 
        expected_output: np.ndarray,
        regression: bool
        ) -> float:
        if regression:
            raise FunctionNotAdvisedWarning('The log loss function is not advised for regression tasks')
        
        raise NotImplementedError()

    @staticmethod
    def __log_loss_derivative(
        weights: np.ndarray, 
        biases: np.ndarray, 
        sample_input: np.ndarray, 
        expected_output: np.ndarray,
        regression: bool
        ) -> float:
        if regression:
            raise FunctionNotAdvisedWarning('The log loss function is not advised for regression tasks')
        
        raise NotImplementedError()

# ----- Activation Functions ----- #

class ActivationFunctions:
    
    @staticmethod
    def hidden(chosen_function: str) -> Callable:
        ''' Provides the chosen activation function
            Options:
            - 'sigmoid': Logistic Sigmoid
            - 'relu': Rectified Linear Unit
            - 'tanh': Hyperbolic Tan
            - 'identity': Passes through the input without applying an activation function
        '''
        match chosen_function:
            case 'sigmoid':
                activation_func = ActivationFunctions.__sigmoid
            case 'tanh':
                activation_func = ActivationFunctions.__tanh
            case 'identity':
                activation_func = ActivationFunctions.__identity
            case 'relu':
                activation_func = ActivationFunctions.__relu
            case _:
                raise ValueError(f'Invalid activation function provided: {chosen_function}.')
        return activation_func

    @staticmethod
    def output(chosen_function: str) -> Callable:
        ''' Provides the chosen output activation function
            Options:
            - 'sigmoid': Logistic Sigmoid (Reccommended for binary or multilabel classification)
            - 'softmax': SoftMax Function (Reccommended for multiclass classification)
            - 'identity': Passes through the input without applying an activation function (Reccommended for regression)
        '''
        match chosen_function:
            case 'sigmoid':
                activation_func = ActivationFunctions.__sigmoid
            case 'softmax':
                activation_func = ActivationFunctions.__softmax
            case 'identity':
                activation_func = ActivationFunctions.__identity
            case _:
                raise ValueError(f'Invalid activation function provided: {chosen_function}.')
        return activation_func

    @staticmethod
    def __softmax(x: np.ndarray) -> np.ndarray:
        ''' Softmax function of a given vector '''
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=1, keepdims=True)

    @staticmethod
    def __sigmoid(x: np.ndarray) -> np.ndarray:
        ''' Logistic sigmoid function of a given vector '''
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def __relu(x: np.ndarray) -> np.ndarray:
        ''' Rectified linear unit function of a given vector '''
        return x * (x > 0)

    @staticmethod
    def __identity(x: np.ndarray) -> np.ndarray:
        ''' Returns the input vector unchanged '''
        return x

    @staticmethod
    def __tanh(x: np.ndarray) -> np.ndarray:
        ''' Tanh of a given vector '''
        return np.tanh(x)

# ----- Weight Initialisation ----- #

class WeightInit:

    @staticmethod
    def generator(chosen_function: str, activation_function: str):
        match chosen_function:
            case 'auto':
                if activation_function in ['tanh', 'sigmoid']:
                    return WeightInit.__norm_xavier
                else:
                    return WeightInit.__he
            case 'he':
                return WeightInit.__he
            case 'xavier':
                return WeightInit.__norm_xavier
            case 'uniform':
                return WeightInit.__uniform
            case _:
                raise ValueError(f'Invalid weight initialisation function provided: {chosen_function}.')

    @staticmethod
    def layer_weights(in_size: int, out_size: int, generator: Callable) -> np.ndarray:
        ''' Generates a weight matrix for one hidden layer '''
        return generator(-1, 1, (in_size, out_size))

    @staticmethod
    def layer_biases(size: int) -> np.ndarray:
        ''' Generates a bias matrix for one hidden layer '''
        return np.zeros((size, 1))

    @staticmethod
    def __uniform(in_size: int, out_size: int) -> np.ndarray:
        ''' Generates a weight matrix uniform on the interval [-1, 1] '''
        return np.random.uniform(-1, 1, (in_size, out_size))

    @staticmethod
    def __norm_xavier(in_size: int, out_size: int) -> np.ndarray:
        ''' Generates a weight matrix according to normalised Xavier initialisation '''
        bound = np.sqrt(6) / np.sqrt(in_size + out_size)
        return np.random.uniform(-bound, bound, (in_size, out_size))

    @staticmethod
    def __he(in_size: int, out_size: int) -> np.ndarray:
        ''' Generates a weight matrix according to He initialisation '''
        std = np.sqrt(in_size)
        return np.random.normal(0, std, (in_size, out_size))