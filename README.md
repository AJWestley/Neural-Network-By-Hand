# Neural Network from Scratch

A feed-forward neural network classification framework I made with just Numpy.
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
    - `'tanh'` (*default*): Hyperbolic Tan
    - `'sigmoid'`: Logistic Sigmoid
    - `'relu'`: Rectified Linear Unit
- **output_activation_function** : string (*optional*)
    Allows the output activation function to be chosen.
    Options:
    - `'softmax'` (*default*): SoftMax Function
    - `'sigmoid'`: Logistic Sigmoid
    - `'relu'`: Rectified Linear Unit
    - `'identity'`: Input Passed Through Unchanged
- **regularisation**: float (*kwarg*)
    The regularisation parameter.
    `0` by default.
- **weight_initialisation**: string (*kwarg*)
    Chooses the distribution with which to initialise the model weights.
    Options:
    - `'auto'` (*default*): A method will be chosen automatically.
    - `'xavier'`: The normalised Xavier method
    - `'he'`: The He method
    - `'uniform'`: Uniform on the interval [-1, 1]
- **learning_rate**: float
    The models rate of learning during each epoch.
    `1e-3` by default.
- **model_type**: string
    The type of model being trained.
    Options:
    - `'auto'` (*default*): Selects the model type automatically.
    - `'multiclass'`: Multiclass classification
    - `'binary'`: Binary or multilabel classification
    - `'regression'`: Regression 

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