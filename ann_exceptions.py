class NotInitialisedError(RuntimeError):
    ''' For errors due to the NeuralNetwork being uninitialised '''
    pass

class FunctionNotAdvisedWarning(RuntimeWarning):
    ''' For when a loss function is being used in a context it is not reccommended for '''
    pass