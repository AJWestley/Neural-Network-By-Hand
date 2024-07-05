import numpy as np
from neural_net_utils import hidden_activation, output_activation, NotInitialisedError
from feed_forward import feed_forward

class NeuralNet:
    ''' A feed-forward neural network '''
    
    def __init__(
        self, 
        hidden_layers: tuple[int] | int, 
        hidden_layer_activation_function: str = 'tanh',
        output_layer_activation_function: str = 'softmax',
        *,
        regression: bool = True
        ) -> None:
        
        self.__topology = __process_hidden_layers(hidden_layers)
        self.__hidden_act = hidden_activation(hidden_layer_activation_function)
        self.__output_act = output_activation(output_layer_activation_function)
        self.__reg = regression
        
        self.weights: tuple | None = None
        self.biases: tuple  | None = None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        ''' Predicts labels for the given data '''
        if self.__reg:
            return self.probabilities(X)
        return np.argmax(self.probabilities(X), axis=1)
    
    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        ''' Initialises the weights and trains the network from scratch '''
        
        _, input_size = X.shape
        
        if self.__reg:
            output_size = 1
        else:
            output_size = len(np.unique(Y, axis=0))
        
        self.initialise_network(input_size, output_size)
        self.add_train(X, Y)
    
    def probabilities(self, X: np.ndarray) -> np.ndarray:
        ''' Runs the feed forward algorithm to get the output probabilities of the network '''
        if self.weights is None or self.biases is None:
            raise NotInitialisedError('Cannot make predictions when the model has not yet been initialised')
        return feed_forward(X, self.weights, self.biases, self.__hidden_act, self.__output_act)
    
    def add_train(self, X: np.ndarray, Y: np.ndarray) -> None:
        ''' Used to continue training when the model is already partially trained '''
        pass
    
    def initialise_network(self, input_size: int, output_size: int) -> None:
        ''' Initialises the weights and biases for the network.
            Is used internally to set up the network.
            
            *Should only be used externally if you wish to experiment with an untrained network.*
        '''
        topology = [input_size] + self.__topology + [output_size]
        
        self.weights = tuple([__init_weights(topology[i-1], topology[i]) for i in range(1, len(topology))])
        self.biases = tuple([___init_biases(topology[i]) for i in range(1, len(topology))])
        

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

def __init_weights(in_size: int, out_size: int) -> np.ndarray:
    ''' Generates a weight matrix for one hidden layer '''
    return np.random.uniform(-1, 1, (in_size, out_size))

def ___init_biases(size: int) -> np.ndarray:
    ''' Generates a bias matrix for one hidden layer '''
    return np.random.uniform(-1, 1, (size, 1))