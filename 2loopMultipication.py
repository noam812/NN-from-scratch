# Define inputs to the neural network
inputs = [1, 2, 3, 2.5]

# Define weights and biases for the neural network layer
# In this example, we have 3 neurons in the layer, and each neuron has 4 weights and a bias term.
# The weights are used to multiply the input values to produce the neuron's weighted sum.
# The biases are added to the weighted sum to produce the neuron's output value.
weights = [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

# Compute the output of the neural network layer
# We iterate through each neuron in the layer, and for each neuron, we compute its weighted sum and add its bias.
# We use a loop to iterate through each input value and weight for the current neuron, and compute the weighted sum.
# Finally, we add the bias term to the weighted sum to produce the neuron's output value.
layer_outputs = [] 
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

# Print the output of the neural network layer
print(layer_outputs)