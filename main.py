import random
import math

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Create a simple dataset
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

# Define the neural network architecture
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1

# Initialize weights and biases with random values
random.seed(1)
weights_input_hidden = [[random.uniform(0, 1) for _ in range(hidden_size)] for _ in range(input_size)]
bias_hidden = [0 for _ in range(hidden_size)]
weights_hidden_output = [random.uniform(0, 1) for _ in range(hidden_size)]
bias_output = 0

# Training loop
for epoch in range(10000):
    # --- Forward Propagation ---
    hidden_input = [0] * hidden_size
    hidden_output = [0] * hidden_size
    
    # Input layer to hidden layer
    for i in range(hidden_size):
        for j in range(input_size):
            hidden_input[i] += X[j][0] * weights_input_hidden[j][i]
        hidden_input[i] += bias_hidden[i]
        hidden_output[i] = sigmoid(hidden_input[i])
    
    # Hidden layer to output layer
    output_input = 0
    for i in range(hidden_size):
        output_input += hidden_output[i] * weights_hidden_output[i]
    output_input += bias_output
    final_output = sigmoid(output_input)
    
    # Calculate the loss
    loss = 0.5 * (final_output - y[0]) ** 2
    
    # --- Backward Propagation ---
    # Calculate gradients
    d_loss = final_output - y[0]
    d_output = d_loss * sigmoid_derivative(final_output)
    d_hidden = [0] * hidden_size
    for i in range(hidden_size):
        d_hidden[i] = d_output * weights_hidden_output[i] * sigmoid_derivative(hidden_output[i])
    
    # Update weights and biases
    for i in range(hidden_size):
        weights_hidden_output[i] -= hidden_output[i] * d_output * learning_rate
    bias_output -= d_output * learning_rate
    for i in range(input_size):
        for j in range(hidden_size):
            weights_input_hidden[i][j] -= X[i][0] * d_hidden[j] * learning_rate
    for i in range(hidden_size):
        bias_hidden[i] -= d_hidden[i] * learning_rate
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# After training, you can use the network for predictions
predicted_output = final_output
print("Predicted Output:")
print(predicted_output)
