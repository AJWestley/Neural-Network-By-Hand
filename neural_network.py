import numpy as np
from utilities import ActivationFunctions, WeightInit
from utilities.ANN_Exeptions import NotInitialisedError
from utilities.FeedForward import feed_forward
from utilities.BackProp import back
from utilities.CostFunctions import CategoricalCrossEntropy

class NeuralNet:
    ''' A feed-forward neural network '''
    
    def __init__(
        self, 
        hidden_layers: tuple[int] | int, 
        hidden_layer_activation_function: str = 'relu',
        output_layer_activation_function: str = 'softmax',
        *,
        weight_initialisation: str = 'auto',
        learning_rate: float = 1e-3,
        regression: bool = False
        ) -> None:
        ''' Constructs a NeuralNet object 
            
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
        
        # Activation Functions
        self.__output_act = ActivationFunctions.output(output_layer_activation_function)
        self.__hidden_act = ActivationFunctions.hidden(hidden_layer_activation_function)
        self.__activation_derivative = ActivationFunctions.derivative(hidden_layer_activation_function)
        
        # Initialise Topology
        self.__topology = NeuralNet.__format_hidden_layers(hidden_layers)
        self.weights: list | None = None
        self.biases: list  | None = None
        
        # Cost Function TODO:
        self.cost = CategoricalCrossEntropy.cost
        self.__hidden_weight_derivative = CategoricalCrossEntropy.hidden_layer_weight_derivative
        self.__hidden_bias_derivative = CategoricalCrossEntropy.hidden_layer_bias_derivative
        self.__output_weight_derivative = CategoricalCrossEntropy.output_layer_weight_derivative
        self.__output_bias_derivative = CategoricalCrossEntropy.output_layer_bias_derivative
        
        # Other Parameters
        self.learning_rate = learning_rate
        self.__reg = regression
        self.__generator = WeightInit.generator(weight_initialisation, hidden_layer_activation_function)
    
    # ----- Training ----- #
    
    def train(self, X: np.ndarray, Y: np.ndarray, *, num_epochs: int = 1000, track_cost: bool = False) -> None:
        ''' Initialises the weights and trains the network from scratch '''
        
        _, input_size = X.shape
        if self.__reg:
            output_size = 1
        else:
            output_size = len(np.unique(Y, axis=0))
        
        self.initialise_network(input_size, output_size)
        self.continue_training(X, Y, num_epochs=num_epochs, track_cost=track_cost)
    
    def continue_training(self, X: np.ndarray, Y: np.ndarray, *, num_epochs: int = 1000, track_cost: bool = False) -> None:
        ''' Used to continue training when the model is already partially trained '''
        
        if self.weights is None or self.biases is None:
            raise NotInitialisedError('Cannot make predictions when the model has not yet been initialised')
        
        for _ in range(num_epochs):
            back(X, Y, self.weights, self.biases, self.learning_rate, self.__hidden_act, self.__output_act, 
                    self.__output_weight_derivative, self.__output_bias_derivative, self.__hidden_weight_derivative,
                    self.__hidden_bias_derivative, self.__activation_derivative, self.cost if track_cost else None)
        
    
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
        
        self.weights = [WeightInit.layer_weights(topology[i-1], topology[i], self.__generator) for i in range(1, len(topology))]
        self.biases = [WeightInit.layer_biases(topology[i]) for i in range(1, len(topology))]

    @staticmethod
    def __format_hidden_layers(hidden_layers: tuple[int] | int) -> list[int]:
        ''' Performs type checking on the topology specification and reformats to a tuple '''
        
        if isinstance(hidden_layers, int):
            hidden_layers = (hidden_layers,)
        
        if not isinstance(hidden_layers, tuple):
            raise TypeError(f"hidden_layers should be of type 'tuple' or 'int'. Provided value is of type '{type(hidden_layers)}'")
        
        for size in hidden_layers:
            if not isinstance(size, int):
                raise TypeError(f"The hidden_layer tuple must contain only elements of type 'int', but an element was found of type '{type(size)}'")
            
        return list(hidden_layers)

