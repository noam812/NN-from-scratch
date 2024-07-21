import numpy as np

# Define output values for three layers
layer_outputs =  [[4.8,1.21,2.385],
                  [8.9,-1.81,0.2],
                  [1.41,1.051,0.026]]

# Apply exponential function to each element of the matrix
# This will make all the values positive and remove negative values
exp_values = np.exp(layer_outputs)

# Normalize the exponential values in order to get the final probability values
# Here, 'axis=1' means we perform sum operation over row axis and 'keepdims=True'
# returns the output in the same dimension as input array, so that we can divide
# the matrix by the sum values.
norm_values = exp_values / np.sum(exp_values, axis=1 , keepdims=True)

# Print the normalized values
print(norm_values)

# Check if the sum of probabilities is 1, which is expected
print(sum(norm_values))


"""
    In the above code, we have applied the softmax activation function to a 3x3 matrix, layer_outputs.
    Softmax function is used to convert the outputs of the previous layer to probability values, so that we can use them in the next layer of the neural network.
    exp_values is the result of applying the exponential function to each element of layer_outputs.
    By using exponential function, we can remove negative values and make all the values positive.
    In order to get the final probability values, we normalize the exp_values using the sum of exp_values along axis=1 (which means the sum is performed row-wise), and keep the dimensions consistent by setting keepdims=True.
    This normalization is important as it scales the probability values between 0 and 1, and ensures that their sum is 1.
    
    The axis parameter is used to specify which axis of the input array should be used to perform the operation.
    Here, we set axis=1 so that we sum across rows.
    
    The keepdims parameter is used to specify whether to keep the dimensions of the input array after the operation or not.
    Here, we set keepdims=True so that the result has the same dimensions as the input array.

    Finally, we print the normalized values and check if the sum of probabilities is 1. 
    If the sum is not 1, then there is a problem with our calculations.

Note that when we use exponential function to compute exp_values, the result may be too large to be represented by a floating point number, causing an overflow. To prevent this, we can use a trick called the log-sum-exp trick. This trick involves taking the log of the numerator and denominator, and then exponentiating the result to get the final probability values. This avoids the exponentiation of large values, and thus prevents overflow.
"""