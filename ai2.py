import numpy as np

class MathNeuralNetwork:
    def __init__(self, input_size=2, hidden_size=8, output_size=1):
        # Initialize weights and biases with random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def forward(self, X):
        """Forward propagation"""
        # First layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # Linear activation for regression
        
        return self.a2
    
    def backward(self, X, y, learning_rate=0.01):
        """Backward propagation"""
        m = X.shape[0]
        
        # Output layer
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update parameters
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        """Train the neural network"""
        losses = []
        
        for epoch in range(epochs):
            # Forward propagation
            predictions = self.forward(X)
            
            # Calculate loss (MSE)
            loss = np.mean((predictions - y) ** 2)
            losses.append(loss)
            
            # Backward propagation
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                
        return losses

# Example usage for addition
def generate_addition_data(num_samples=1000):
    """Generate training data for addition"""
    X = np.random.uniform(0, 10, (num_samples, 2))
    y = np.sum(X, axis=1, keepdims=True)
    return X, y

# Create and train the network
if __name__ == "__main__":
    # Generate training data
    X_train, y_train = generate_addition_data(1000)
    
    # Initialize and train the network
    nn = MathNeuralNetwork(input_size=2, hidden_size=8, output_size=1)
    losses = nn.train(X_train, y_train, epochs=50000, learning_rate=0.00001)
    
    # Test the network
    test_input = np.array([[5.0, 3.0]])
    prediction = nn.forward(test_input)
    print(f"\nTest: {test_input[0,0]} + {test_input[0,1]} = {prediction[0,0]:.2f}")