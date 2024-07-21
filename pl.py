import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# initialize nnfs library
nnfs.init()

# define Layer_Dense class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # initialize weights with Gaussian distribution (mean 0, variance 1) scaled by 0.01
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # initialize biases with zeros
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        # calculate output of layer by multiplying inputs by weights, adding biases, and passing through activation function
        self.output = np.dot(inputs, self.weights) + self.biases

# define Activation_ReLU class
class Activation_ReLU:
    def forward(self, inputs):
        # apply ReLU activation function to inputs (i.e. set all negative values to 0)
        self.output = np.maximum(0, inputs)
        
# define Activation_Softmax class
class Activation_Softmax:
    def forward(self, inputs):
        # subtract maximum value from inputs to prevent overflow when exponentiating
        exp_values = np.exp(inputs - np.max(inputs ,axis=1 , keepdims=True))
        # normalize exponentiated values to get probabilities
        probabilities = exp_values / np.sum(exp_values , axis=1 ,keepdims=True)
        self.output = probabilities

# define Loss class
class Loss:
    #output is the output from the model for loss calc 
    #y for the intented target  training values
    def calculate(self,output,y):
        # calculate the average loss for a batch of samples 
        #forward will vary based on the type of calculation loss 
        #in this case @Loss_CategoricalCrossentropy
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
# define Loss_CategoricalCrossentropy class that inherits from Loss class
class Loss_CategoricalCrossentropy(Loss):
    #y_pred the model output    
    #y_true the target training
        # calculate the categorical cross-entropy loss for a batch of samples
    def forward(self, y_pred, y_true):
        samples =len(y_pred)
        # clip the predicted values to avoid log(0) or log(1) errors
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        # its good parctice to prepare loss functions for both scanarios training data
        if len(y_true.shape) ==1:
            # if the true labels are one-dimensional (i.e. sparse), use them as indices to select the correct predicted probabilities
            correct_confidences =  y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape)==2:
            # if the true labels are two-dimensional (i.e. one-hot encoded), multiply them element-wise with the predicted probabilities and sum along the second axis
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # take the negative logarithm of the correct predicted probabilities and return them as the loss values
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    

    

# generate spiral data
X, y = spiral_data(samples=100 , classes=3)
print(y)

# create first dense layer with 2 inputs and 3 neurons
dense1 = Layer_Dense(2, 3)
# create ReLU activation function object
act1 = Activation_ReLU()

# create second dense layer with 3 inputs and 3 neurons
dense2 = Layer_Dense(3, 3)
# create Softmax activation function object
act2 = Activation_Softmax()

# pass input data through first dense layer and ReLU activation function
dense1.forward(X)
act1.forward(dense1.output)

# pass output of first layer through second dense layer and Softmax activation function
dense2.forward(act1.output)
act2.forward(dense2.output)

# print first 5 probabilities of the output of the second layer
print(act2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(act2.output , y )

print(loss)