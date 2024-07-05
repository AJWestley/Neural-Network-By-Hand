from typing import Callable
import numpy as np
from utilities.ANN_Exeptions import FunctionNotAdvisedWarning

def cost_function(chosen_function: str) -> Callable:
    raise NotImplementedError()

def cost_derivative(chosen_function: str) -> Callable:
    raise NotImplementedError()

# For Regression

def __mean_squared_error(
    prediction: np.ndarray,
    expected_output: np.ndarray,
    regression: bool
    ) -> float:
    if not regression:
        raise FunctionNotAdvisedWarning('The mean squared error function is not advised for classification tasks')
    n, _ = expected_output.shape
    raise NotImplementedError()

def __mean_squared_error_derivative(
    prediction: np.ndarray,
    expected_output: np.ndarray,
    regression: bool
    ) -> float:
    if not regression:
        raise FunctionNotAdvisedWarning('The mean squared error function is not advised for classification tasks')
    
    raise NotImplementedError()

def __mean_absolute_error(
    prediction: np.ndarray,
    expected_output: np.ndarray,
    regression: bool
    ) -> float:
    if not regression:
        raise FunctionNotAdvisedWarning('The mean absolute error function is not advised for classification tasks')
    
    raise NotImplementedError()

def __mean_absolute_error_derivative(
    prediction: np.ndarray,
    expected_output: np.ndarray,
    regression: bool
    ) -> float:
    if not regression:
        raise FunctionNotAdvisedWarning('The mean squared error function is not advised for classification tasks')
    
    raise NotImplementedError()

def __huber_loss(
    prediction: np.ndarray,
    expected_output: np.ndarray,
    regression: bool
    ) -> float:
    if not regression:
        raise FunctionNotAdvisedWarning('The huber loss function is not advised for classification tasks')
    
    raise NotImplementedError()

def __huber_loss_derivative(
    prediction: np.ndarray,
    expected_output: np.ndarray,
    regression: bool
    ) -> float:
    if not regression:
        raise FunctionNotAdvisedWarning('The huber loss function is not advised for classification tasks')
    
    raise NotImplementedError()

# For Classification

def __binary_cross_entropy(
    prediction: np.ndarray,
    expected_output: np.ndarray,
    regression: bool
    ) -> float:
    if regression:
        raise FunctionNotAdvisedWarning('The binary cross entropy function is not advised for regression tasks')
    
    raise NotImplementedError()

def __binary_cross_entropy_derivative(
    prediction: np.ndarray,
    expected_output: np.ndarray,
    regression: bool
    ) -> float:
    if regression:
        raise FunctionNotAdvisedWarning('The binary cross entropy function is not advised for regression tasks')
    
    raise NotImplementedError()

def __categorical_cross_entropy(
    prediction: np.ndarray,
    expected_output: np.ndarray,
    regression: bool
    ) -> float:
    if regression:
        raise FunctionNotAdvisedWarning('The categorical cross entropy function is not advised for regression tasks')
    
    raise NotImplementedError()

def __categorical_cross_entropy_derivative(
    prediction: np.ndarray,
    expected_output: np.ndarray,
    regression: bool
    ) -> float:
    if regression:
        raise FunctionNotAdvisedWarning('The categorical cross entropy function is not advised for regression tasks')
    
    raise NotImplementedError()

def __hinge_loss(
    prediction: np.ndarray,
    expected_output: np.ndarray,
    regression: bool
    ) -> float:
    if regression:
        raise FunctionNotAdvisedWarning('The hinge loss function is not advised for regression tasks')
    
    raise NotImplementedError()

def __hinge_loss_derivative(
    prediction: np.ndarray,
    expected_output: np.ndarray,
    regression: bool
    ) -> float:
    if regression:
        raise FunctionNotAdvisedWarning('The chinge loss function is not advised for regression tasks')
    
    raise NotImplementedError()

def __log_loss(
    prediction: np.ndarray,
    expected_output: np.ndarray,
    regression: bool
    ) -> float:
    if regression:
        raise FunctionNotAdvisedWarning('The log loss function is not advised for regression tasks')
    
    raise NotImplementedError()

def __log_loss_derivative(
    prediction: np.ndarray,
    expected_output: np.ndarray,
    regression: bool
    ) -> float:
    if regression:
        raise FunctionNotAdvisedWarning('The log loss function is not advised for regression tasks')
    
    raise NotImplementedError()