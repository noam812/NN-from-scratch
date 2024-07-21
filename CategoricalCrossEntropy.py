import math

# Softmax probabilities
softmax_output = [0.7, 0.1, 0.2]

# Target probabilities (one-hot encoded)
target_output = [1, 0, 0]

# Categorical Cross-Entropy Loss Calculation
# -(t1*log(p1) + t2*log(p2) + t3*log(p3))
# where t1,t2,t3 are target output probabilities and p1,p2,p3 are softmax output probabilities
loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2] )

# Print the calculated loss
print(loss)

# Categorical Cross-Entropy Loss Calculation with only the first output
# -log(p1) where p1 is softmax output probabilities for the first class
loss2 = -math.log(softmax_output[0])
print(loss2)

"""_summary_
The code calculates the Categorical Cross-Entropy Loss between the Softmax output probabilities and the target probabilities.

The Softmax output probabilities represent the predicted probabilities for each class, while the target probabilities represent the true probabilities for each class.

The loss calculation is done by summing the negative log-likelihood of each class's predicted probability, weighted by the true probability for that class. This loss calculation penalizes the model more heavily for making incorrect predictions with higher confidence.

The first calculation, loss, calculates the loss for all three classes. It multiplies the negative natural logarithm of each predicted probability by its corresponding target probability, and then sums them up. This gives a single number that represents the overall loss.

The second calculation, loss2, calculates the loss for only the first class. It does this by simply taking the negative natural logarithm of the first predicted probability.

Both of these calculations output a scalar value representing the loss.
"""