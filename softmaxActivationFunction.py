import math

# Define the output of the neural network's layer
layer_outputs =  [4.8,1.21,2.385]

# Define the Euler's number ( 2.71828182846)
E = math.e

# Create an empty list to hold the exponential values for each output
exp_values = []

# Calculate the exponential value for each output and add it to the list
for output in layer_outputs:
    exp_values.append(E**output)

# Print the list of exponential values
print(exp_values)    

# Calculate the normalization constant (the sum of all exponential values)
norm_base = sum(exp_values)

# Create an empty list to hold the normalized values
norm_values = []

# Divide each exponential value by the normalization constant and add it to the list
for value in exp_values:
    norm_values.append(value / norm_base)

# Print the list of normalized values
print(norm_values)

# Verify that the normalized values sum up to 1 (within a small margin of error)
print(sum(norm_values))