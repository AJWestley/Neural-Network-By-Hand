from typing import Callable
import numpy as np
from utilities.ANN_Exeptions import FunctionNotAdvisedWarning

def cost_function(chosen_function: str, regression: bool) -> Callable:
    ''' Acts like a cost function dictionary with detailed errors '''
    
    match chosen_function:
        case 'MSE':
            if not regression:
                raise FunctionNotAdvisedWarning('The mean squared error function is not advised for classification tasks')
            return __mean_squared_error
        case 'MAE':
            if not regression:
                raise FunctionNotAdvisedWarning('The mean absolute error function is not advised for classification tasks')
            return __mean_absolute_error
        case 'huber':
            if not regression:
                raise FunctionNotAdvisedWarning('The huber loss function is not advised for classification tasks')
            return __huber_loss
        case 'bin_cross':
            if regression:
                raise FunctionNotAdvisedWarning('The binary cross entropy function is not advised for regression tasks')
            return __binary_cross_entropy
        case 'cat_cross':
            if regression:
                raise FunctionNotAdvisedWarning('The categorical cross entropy function is not advised for regression tasks')
            return __categorical_cross_entropy
        case 'hinge':
            if regression:
                raise FunctionNotAdvisedWarning('The hinge loss function is not advised for regression tasks')
            return __hinge_loss
        case 'log':
            if regression:
                raise FunctionNotAdvisedWarning('The log loss function is not advised for regression tasks')
            return __log_loss
        case _:
            raise ValueError(f'Invalid cost function provided: {chosen_function}.')

def cost_derivative(chosen_function: str, regression: bool) -> Callable:
    ''' Acts like a cost function derivative dictionary with detailed errors '''
    
    match chosen_function:
        case 'MSE':
            if not regression:
                raise FunctionNotAdvisedWarning('The mean squared error function is not advised for classification tasks')
            return __mean_squared_error_derivative
        case 'MAE':
            if not regression:
                raise FunctionNotAdvisedWarning('The mean absolute error function is not advised for classification tasks')
            return __mean_absolute_error_derivative
        case 'huber':
            if not regression:
                raise FunctionNotAdvisedWarning('The huber loss function is not advised for classification tasks')
            return __huber_loss_derivative
        case 'bin_cross':
            if regression:
                raise FunctionNotAdvisedWarning('The binary cross entropy function is not advised for regression tasks')
            return __binary_cross_entropy_derivative
        case 'cat_cross':
            if regression:
                raise FunctionNotAdvisedWarning('The categorical cross entropy function is not advised for regression tasks')
            return __categorical_cross_entropy_derivative
        case 'hinge':
            if regression:
                raise FunctionNotAdvisedWarning('The hinge loss function is not advised for regression tasks')
            return __hinge_loss_derivative
        case 'log':
            if regression:
                raise FunctionNotAdvisedWarning('The log loss function is not advised for regression tasks')
            return __log_loss_derivative
        case _:
            raise ValueError(f'Invalid cost function provided: {chosen_function}.')

# For Regression

def __mean_squared_error(
    prediction: np.ndarray,
    expected_output: np.ndarray
    ) -> float:
    
    n, _ = expected_output.shape
    raise NotImplementedError()

def __mean_squared_error_derivative(
    prediction: np.ndarray,
    expected_output: np.ndarray
    ) -> float:
    
    raise NotImplementedError()

def __mean_absolute_error(
    prediction: np.ndarray,
    expected_output: np.ndarray
    ) -> float:
    
    raise NotImplementedError()

def __mean_absolute_error_derivative(
    prediction: np.ndarray,
    expected_output: np.ndarray
    ) -> float:
    
    raise NotImplementedError()

def __huber_loss(
    prediction: np.ndarray,
    expected_output: np.ndarray
    ) -> float:
    
    raise NotImplementedError()

def __huber_loss_derivative(
    prediction: np.ndarray,
    expected_output: np.ndarray
    ) -> float:
    
    raise NotImplementedError()

# For Classification

def __binary_cross_entropy(
    prediction: np.ndarray,
    expected_output: np.ndarray
    ) -> float:
    
    raise NotImplementedError()

def __binary_cross_entropy_derivative(
    prediction: np.ndarray,
    expected_output: np.ndarray
    ) -> float:
    
    raise NotImplementedError()

def __categorical_cross_entropy(
    prediction: np.ndarray,
    expected_output: np.ndarray
    ) -> float:
    
    raise NotImplementedError()

def __categorical_cross_entropy_derivative(
    prediction: np.ndarray,
    expected_output: np.ndarray
    ) -> float:
    
    raise NotImplementedError()

def __hinge_loss(
    prediction: np.ndarray,
    expected_output: np.ndarray
    ) -> float:
    
    raise NotImplementedError()

def __hinge_loss_derivative(
    prediction: np.ndarray,
    expected_output: np.ndarray
    ) -> float:
    
    raise NotImplementedError()

def __log_loss(
    prediction: np.ndarray,
    expected_output: np.ndarray
    ) -> float:
    
    raise NotImplementedError()

def __log_loss_derivative(
    prediction: np.ndarray,
    expected_output: np.ndarray
    ) -> float:
    
    raise NotImplementedError()