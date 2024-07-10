from sys import argv
from matplotlib import pyplot as plt
import numpy as np
from neural_network import NeuralNet

def plot2D(X, Y, T, epoch):
    plt.clf()
    plt.title(f"Epoch: {epoch}")
    plt.plot(X, Y, c='red')
    plt.plot(X, T, c='blue')
    plt.legend(['Estimation', 'True Function'])
    plt.pause(0.1)

def sin_animation(ep_size = 500):
    
    X = np.linspace(0, 4 * np.pi, 100).reshape((-1, 1))
    T = np.sin(X)
    
    ann = NeuralNet((50, 50, 50), output_activation_function='identity', learning_rate=1e-4, regularisation=1)
    
    ann.train(X, T, num_epochs=0)
    
    Y = ann.predict(X)
    plot2D(X, Y, T, 0)
    
    for i in range(1, 101):
        epoch = i * ep_size
        ann.continue_training(X, T, num_epochs=ep_size)
        Y = ann.predict(X)
        plot2D(X, Y, T, epoch)
    plt.show()

def abs_animation(ep_size = 500):
    X = np.linspace(-1, 1, 100).reshape((-1, 1))
    T = np.abs(X)
    
    ann = NeuralNet((50, 50, 50, 50), output_activation_function='relu', learning_rate=1e-4)
    
    ann.train(X, T, num_epochs=0)
    
    Y = ann.predict(X)
    plot2D(X, Y, T, 0)
    
    for i in range(1, 101):
        epoch = i * ep_size
        ann.continue_training(X, T, num_epochs=ep_size)
        Y = ann.predict(X)
        plot2D(X, Y, T, epoch)
    plt.show()

if __name__ == '__main__':
    anim = argv[1]
    
    match anim:
        case 'sin':
            sin_animation()
        case 'abs':
            abs_animation()
        case _:
            print('Please choose either "sin" or "abs"')