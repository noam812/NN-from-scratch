import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# initialize nnfs library
nnfs.init()

# generate spiral data for testing
#X, y = spiral_data(100, 3)

# input data with 3 training examples and 4 features each
X = [[1, 2, 3, 2.5],
      [2, 5, -1, 2],
      [-1.5, 2.7, 3.3, -0.8]]

# Class for creating dense neural network layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # initialize weights randomly with shape (n_inputs, n_neurons)
        # using Gaussian distribution scaled down by a factor of 0.1
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        
        # initialize biases to zero with shape (1, n_neurons)
        # avoiding initializing to zero is important to prevent "neural network death"
        self.biases = np.zeros((1, n_neurons))

    # calculate output for the layer
    def forward(self, inputs):
        # dot product of input and weights plus biases
        self.output = np.dot(inputs, self.weights) + self.biases

# Class for applying Rectified Linear Unit (ReLU) activation function
class Activation_ReLU:
    def forward(self, inputs):
        # apply ReLU activation function
        self.output = np.maximum(0, inputs)

# create dense neural network layers with 5 neurons in the first layer and 3 in the second
layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 3)

# apply forward propagation for each layer
layer1.forward(X)
layer2.forward(layer1.output)

# print the output of the second layer
print(layer2.output)
