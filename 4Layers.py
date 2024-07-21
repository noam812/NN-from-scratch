import numpy as np
#This neural network has an input layer with 4 neurons, which means that the input data has 4 features.
#Since there are 3 training examples, the shape of the input data is (3, 4).
#The hidden layer has 3 output neurons.
#This is represented by the number of values in the biases list, which is also 3.
#The number of columns in the weights matrix represents the number of neurons in the hidden layer, which is also 3 in this case.



# Define inputs, weights, and biases for the first layer
inputs = [[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-0.8]]
weights = [[0.2,0.8,-0.5,1],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]
biases = [2,3,0.5]

# Define weights and biases for the second layer
weights2 = [[0.1,-0.14,0.5],[-0.5, 0.12, -0.33],[-0.44,0.73,-0.13]]
biases2 = [-1,2,-0.5]

# First layer:
# The shape of the weight matrix connecting the inputs to the first layer depends on the number of neurons in each layer.
# In this code, the first layer has 4 input neurons and 3 output neurons.
# So the weight matrix connecting these two layers has a shape of (3,4).
# The shape of the bias vector for the first layer has to be equal to the number of output neurons in that layer, which is 3 in this case.
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
print("Layer 1 outputs:", layer1_outputs)

# Second layer:
# The shape of the weight matrix connecting the first layer to the second layer also depends on the number of neurons in each layer.
# In this code, the second layer has 3 input neurons and 3 output neurons.
# So the weight matrix connecting these two layers has a shape of (3,3).
# The shape of the bias vector for the second layer has to be equal to the number of output neurons in that layer, which is 3 in this case.
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print("Layer 2 outputs:", layer2_outputs)
