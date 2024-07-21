import numpy as np

# Explanation of the transpose of a matrix and how to get it in numpy
# The transpose of a matrix is a new matrix whose rows are the columns of the original matrix and whose columns are the rows of the original matrix.
# In numpy, you can use the .T attribute to get the transpose of a matrix.

# Define a 2x2 matrix using numpy
matrix = np.array([[1, 2], [3, 4]])

# Print the original matrix
print("Original Matrix:\n", matrix)
#1,2
#3,4


# Print the transpose of the matrix
print("Transpose of Matrix:\n", matrix.T)
#1,3
#2,4


# Define inputs, weights, and biases for the neural network
inputs = [[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-0.8]]
weights = [[0.2,0.8,-0.5,1],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]
biases = [2,3,0.5]

# Explanation of the dot product and how it's used in neural networks
# The dot product is a mathematical operation that takes two equal-length vectors and returns a scalar.
# In neural networks, we use the dot product to multiply the inputs with their corresponding weights, summing them up, and adding biases.

# Perform the dot product between the inputs and the transpose of the weights matrix, then add the biases
output = np.dot(inputs,np.array(weights).T) + biases

# Print the output of the neural network
print(output)
