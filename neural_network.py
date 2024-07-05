import numpy as np
from ann_exceptions import NotInitialisedError
from feed_forward import feed_forward
from ann_utils import ActivationFunctions, WeightInit

class NeuralNet:
    ''' A feed-forward neural network '''
    
    def __init__(
        self, 
        hidden_layers: tuple[int] | int, 
        hidden_layer_activation_function: str = 'relu',
        output_layer_activation_function: str = 'softmax',
        *,
        weight_initialisation: str = 'auto',
        regression: bool = True
        ) -> None:
        ''' Constructs a NeuralNetwork object 
            
            Params:
            ---------
            hidden_layers : tuple[int] | int
                A tuple containing the number of neurons to be in each layer (or an int if there is to be one layer)
            
            hidden_layer_activation_function : str
            The activation function to be used in hidden layer neurons.
            
            Options:
            - 'sigmoid': Logistic Sigmoid
            - 'relu': Rectified Linear Unit
            - 'tanh': Hyperbolic Tan
            - 'identity': Passes through the input without applying an activation function
            
            output_layer_activation_function : str
            The activation function to be used in output neurons.
            
            Options:
            - 'sigmoid': Logistic Sigmoid (Reccommended for binary or multilabel classification)
            - 'softmax': SoftMax Function (Reccommended for multiclass classification)
            - 'identity': Passes through the input without applying an activation function (Reccommended for regression)
            
            weight_initialisation : str
            The method by which to initialise the weights.
            
            Options:
            - 'auto': A method will be chosen automatically.
            - 'xavier': The normalised Xavier method
            - 'he': The He method
            - 'uniform': Uniform on the interval [-1, 1]
            
            regression: bool
            Whether or not the network will be used for regression.
        '''
        
        self.__topology = __process_hidden_layers(hidden_layers)
        self.__hidden_act = ActivationFunctions.hidden(hidden_layer_activation_function)
        self.__output_act = ActivationFunctions.output(output_layer_activation_function)
        self.__generator = WeightInit.generator(weight_initialisation, hidden_layer_activation_function)
        self.__reg = regression
        
        self.weights: tuple | None = None
        self.biases: tuple  | None = None
    
    # ----- Training ----- #
    
    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        ''' Initialises the weights and trains the network from scratch '''
        
        _, input_size = X.shape
        if self.__reg:
            output_size = 1
        else:
            output_size = len(np.unique(Y, axis=0))
        
        self.initialise_network(input_size, output_size)
        self.continue_training(X, Y)
    
    # TODO:
    def continue_training(self, X: np.ndarray, Y: np.ndarray) -> None:
        ''' Used to continue training when the model is already partially trained '''
        pass
    
    # ----- Predicting ----- #
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        ''' Predicts labels for the given data '''
        
        if self.__reg:
            return self.probabilities(X)
        return np.argmax(self.probabilities(X), axis=1)
    
    def probabilities(self, X: np.ndarray) -> np.ndarray:
        ''' Runs the feed forward algorithm to get the output probabilities of the network '''
        
        if self.weights is None or self.biases is None:
            raise NotInitialisedError('Cannot make predictions when the model has not yet been initialised')
        
        return feed_forward(X, self.weights, self.biases, self.__hidden_act, self.__output_act)
    
    # ----- Initialisation ----- #
    
    def initialise_network(self, input_size: int, output_size: int) -> None:
        ''' Initialises the weights and biases for the network.
            Is used internally to set up the network.
            
            *Should only be used externally if you wish to experiment with an untrained network.*
        '''
        
        topology = [input_size] + self.__topology + [output_size]
        
        self.weights = tuple([WeightInit.layer_weights(topology[i-1], topology[i], self.__generator) for i in range(1, len(topology))])
        self.biases = tuple([WeightInit.layer_biases(topology[i]) for i in range(1, len(topology))])


def __process_hidden_layers(hidden_layers: tuple[int] | int) -> list[int]:
    ''' Performs type checking on the topology specification and reformats to a tuple '''
    
    if isinstance(hidden_layers, int):
        hidden_layers = (hidden_layers,)
    
    if not isinstance(hidden_layers, tuple):
        raise TypeError(f"hidden_layers should be of type 'tuple' or 'int'. Provided value is of type '{type(hidden_layers)}'")
    
    for size in hidden_layers:
        if not isinstance(size, int):
            raise TypeError(f"The hidden_layer tuple must contain only elements of type 'int', but an element was found of type '{type(size)}'")
        
    return list(hidden_layers)

