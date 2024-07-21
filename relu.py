import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# initialize NNFS library
nnfs.init()

# generate spiral data for classification problem
X, y = spiral_data(100, 3)

# class representing a dense layer of neurons in a neural network
class Layer_Dense:
    # constructor function with two input parameters: number of inputs and number of neurons in the layer
    def __init__(self, n_inputs, n_neurons):
        # initialize weights randomly scaled by 0.1 with shape (n_inputs, n_neurons)
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # initialize biases with zeros with shape (1, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    # forward function with one input parameter: input data for the layer
    def forward(self, inputs):
        # calculate dot product of inputs and weights, add biases, and store result in output attribute
        self.output = np.dot(inputs, self.weights) + self.biases

# class representing a ReLU activation function
class Activation_ReLU:
    # forward function with one input parameter: input data for the activation function
    def forward(self, inputs):
        # apply element-wise ReLU activation function to inputs and store result in output attribute
        self.output = np.maximum(0, inputs)

# create a dense layer with 5 neurons and 2 inputs
layer1 = Layer_Dense(2, 5)
# calculate output of layer1 for the input data X
layer1.forward(X)
# create an instance of ReLU activation function
activation1 = Activation_ReLU()
# apply ReLU activation function to the output of layer1 and store result in output attribute of activation1
activation1.forward(layer1.output)

# print output of layer1 and output of activation1
print(layer1.output)
print(activation1.output)

# define input data for testing the ReLU activation function
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

# apply element-wise ReLU activation function to input data and store result in output list
for i in inputs:
    output.append(max(0, i))

# print output list
print(output)
