# Neural Network from Scratch

A feed-forward neural network I made with just Numpy.
I made this as a means of sinking my teeth into deep learning before learning any APIs.

## Usage

To use the NeuralNet class, it must first be imported as follows:
```python
from neural_network import NeuralNet
```

### The Constructor

##### Parameters:
- **hidden_layers** : tuple
    Specifies how many neurons to be in each hidden layer.
- **hidden_activation_function** : string (*optional*)
    Allows the activation function of the hidden layers to be chosen.
    Options:
    - `'sigmoid'` (*default*): Logistic Sigmoid
    - `'relu'`: Rectified Linear Unit
    - `'tanh'`: Hyperbolic Tan
- **output_activation_function** : string (*optional*)
    Allows the output activation function to be chosen.
    Options:
    - `'softmax'` (*default*): SoftMax Function
    - `'sigmoid'`: Logistic Sigmoid
    - `'relu'`: Rectified Linear Unit
    - `'identity'`: Input Passed Through Unchanged
- **weight_initialisation**: string (*kwarg*)
    Chooses the distribution with which to initialise the model weights.
    Options:
    - `'auto'` (*default*): A method will be chosen automatically.
    - `'xavier'`: The normalised Xavier method
    - `'he'`: The He method
    - `'uniform'`: Uniform on the interval [-1, 1]

- **cost_func**: string
    The cost function to optimise.
    
    Options:
    - `'auto'` (*default*): A function will be chosen automatically.
    - `'bce'`: Binary Cross Entropy (**Not Yet Supported**)
    - `'cce'`: Categorical Cross Entropy
    - `'mse'`: Mean Squared Error (**Not Yet Supported**)

- **learning_rate**: float
    The models rate of learning during each epoch.
    `1e-3` by default.

- **regression**: boolean
    Whether or not the network will be used for regression.
    `False` by default.

##### Example:

For a simple multi-classification problem with a tanh activation function, we can construct our ANN as follows:
```python
ann = NeuralNet((5, 6, 3), 'tanh')
```

### Training the Network

The network can be trained from scratch with the `train` function, or if the model is already partially trained, the `continue_training` can be used. Both of which have the same parameters.

##### Parameters:

- **X**: numpy array
    The input data.
- **Y**: numpy array
    The data labels for X.
- **num_epochs**: integer (*kwarg*)
    How many epochs to train for.
    `1000` by default.

##### Example

The network can be trained as follows:
```python
ann.train(X, Y)
```

### Making Predictions

To make predictions with the model, the `predict` function can be used. To get the output probabilities instead, you can use the `probabilities` function. Both take the same parameter.

##### Parameters:

- **X**: numpy array
    The data from which to make a prediction or calculate the probabilities.

##### Returns:
    A numpy array containing the predictions.

##### Example:

To predict from a dataset X:
```python
predictions = ann.predict(X)
```