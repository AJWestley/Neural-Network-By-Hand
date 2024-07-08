import numpy as np
from utilities import ActivationFunctions, WeightInit
from utilities.ANN_Exeptions import NotInitialisedError
from utilities.FeedForward import feed_forward
from utilities.BackPropagation import back

class NeuralNet:
    ''' A feed-forward neural network '''
    
    def __init__(
        self, 
        hidden_layers: tuple[int] | int, 
        hidden_activation_function: str = 'sigmoid',
        output_activation_function: str = 'softmax',
        *,
        weight_initialisation: str = 'auto',
        learning_rate: float = 1e-3,
        model_type: str = 'auto'
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
            
            output_layer_activation_function : str
            The activation function to be used in output neurons.
            
            Options:
            - 'sigmoid': Logistic Sigmoid (Reccommended for binary or multilabel classification)
            - 'softmax': SoftMax Function (Reccommended for multiclass classification)
            
            weight_initialisation : str
            The method by which to initialise the weights.
            
            Options:
            - 'auto': A method will be chosen automatically.
            - 'xavier': The normalised Xavier method
            - 'he': The He method
            - 'uniform': Uniform on the interval [-1, 1]
            
            learning_rate : float
            The models rate of learning during each epoch.
            
        '''
        
        # Activation Functions
        self.__output_act = ActivationFunctions.output(output_activation_function)
        self.__hidden_act = ActivationFunctions.hidden(hidden_activation_function)
        self.__activation_derivative = ActivationFunctions.derivative(hidden_activation_function)
        
        # Initialise Topology
        self.__topology = NeuralNet.__format_hidden_layers(hidden_layers)
        self.weights: list | None = None
        self.biases: list  | None = None
        
        # Other Parameters
        self.learning_rate = learning_rate
        self.__type = NeuralNet.__set_model_type(model_type, output_activation_function)
        self.__generator = WeightInit.generator(weight_initialisation, hidden_activation_function)

    # ----- Training ----- #

    def train(self, X: np.ndarray, Y: np.ndarray, *, num_epochs: int = 1000) -> None:
        ''' Initialises the weights and trains the network from scratch '''
        
        _, input_size = X.shape
        output_size = Y.shape[1] if len(Y.shape) > 1 else 1
        
        self.initialise_network(input_size, output_size)
        self.continue_training(X, Y, num_epochs=num_epochs)

    def continue_training(self, X: np.ndarray, Y: np.ndarray, *, num_epochs: int = 1000) -> None:
        ''' Used to continue training when the model is already partially trained '''
        
        if self.weights is None or self.biases is None:
            raise NotInitialisedError('Cannot make predictions when the model has not yet been initialised')
        
        for _ in range(num_epochs):
            back(X, Y, self.weights, self.biases, self.learning_rate, self.__hidden_act, 
                self.__output_act, self.__activation_derivative)


    # ----- Predicting ----- #

    def predict(self, X: np.ndarray) -> np.ndarray:
        ''' Predicts labels for the given data '''
        
        if self.__type == 'binary':
            return np.round(self.probabilities(X)).astype(np.uint32)
        if self.__type == 'multiclass':
            return np.argmax(self.probabilities(X), axis=1)
        return self.probabilities(X)

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

    # ----- Utility Methods ----- #
    
    def get_type(self):
        return self.__type

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

    @staticmethod
    def __set_model_type(type: str, output_act: str):
        if type ==  'auto':
                if output_act == 'softmax':
                    return 'multiclass'
                if output_act == 'sigmoid':
                    return 'binary'
                return 'regression'
            
        if type in ['multiclass', 'binary', 'regression']:
            return type
        
        raise ValueError(f"Invalid model type: {type}.")