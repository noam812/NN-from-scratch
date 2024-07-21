# Define parameters
# Below we coded a layer of 4 input and and 3 outputs
# input are actually the data, it might be the raw data or the previous layer data
# so we will not change them.
inputs = [1,2,3,2.5]

# Weights are one of the ways to adjust the data multiplication so that we can get different outcomes.
# In this example, we have 3 neurons in the output layer, each with 4 input weights (one for each input feature)
# These weights are learned during training and are adjusted to minimize the error between the model's predictions and the true outputs.
weights1 = [0.2,0.8,-0.5,1]
weights2 = [0.5,-0.91,0.26,-0.5]
weights3 = [-0.26,-0.27,0.17,0.87]

# Bias is a constant term added to each neuron's weighted sum to shift the activation function left or right.
# It can help the network learn the correct function more easily.
# In this case, we have a bias term for each of the 3 output neurons.
bias1 = 2
bias2 = 3
bias3 = 0.5

# The first step in computing the output of the network is to add up all the weighted inputs for each neuron, and then add the bias term.
# This is known as the weighted sum or "net input".
# In this example, we have 3 neurons in the output layer, so we compute 3 separate weighted sums.
# The output of each neuron is then obtained by passing the weighted sum through an activation function.
# Here, we use a simple linear activation function that just passes through the weighted sum unchanged.
# In practice, other activation functions such as sigmoid, ReLU, or softmax may be used, depending on the problem.
output =[inputs[0]*weights1[0] +inputs[1]*weights1[1] +inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
         inputs[0]*weights2[0] +inputs[1]*weights2[1] +inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
         inputs[0]*weights3[0] +inputs[1]*weights3[1] +inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3] 

# The final output of the network is simply the activation of each neuron in the output layer.
# In this case, since we used a linear activation function, the output is just the same as the weighted sum.
# But if we had used a different activation function, the output would have been transformed accordingly.
# The output is then printed to the console.
print(output)