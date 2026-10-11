import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the model
def create_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(128, activation='relu'),  # Increased neurons
        layers.Dense(128, activation='relu'),
        layers.Dense(1)  # Output layer for derivative
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

    # Generate synthetic training data
def generate_data(num_samples):
    X = np.random.uniform(-10, 10, size=(num_samples, 1))  # Random input values
    a = np.random.uniform(-1, 1, size=num_samples)
    b = np.random.uniform(-1, 1, size=num_samples)
    c = np.random.uniform(-1, 1, size=num_samples)

    y = a * X**2 + b * X + c  # f(x)
    dy_dx = 2 * a * X + b  # f'(x)

    # Include higher-order terms in the input
    X_aug = np.concatenate([X, X**2, X**3], axis=1)  # Add x^2, x^3 as features
    return X_aug, dy_dx

# Increase sample size to 10,000
num_samples = 10000
X_train, y_train = generate_data(num_samples)
print(f"Sample inputs: {X_train}")
print(f"Sample derivatives: {y_train}")


# Create the model
model = create_model(input_shape=(3,))

# Train the model again
model.fit(X_train, y_train, epochs=1000, batch_size=32)

# Function to predict the derivative given a value
def predict_derivative(model, value):
    value = np.array(value).reshape(1, -1)  # Reshape for prediction
    return model.predict(value)[0][0]

def augment_input(value):
    aug_value = np.concatenate([value, value**2, value**3], axis=1)
    print(f"Augmented input for x = {value}: {aug_value}")  # Debugging
    return aug_value

# Example usage
test_value = np.array([[2.0]])  # Test at x = 2.0
test_value = augment_input(test_value)
predicted_derivative = predict_derivative(model, test_value)
print(f"Predicted derivative at x = 2.0: {predicted_derivative}")

# For comparison, calculate the true derivative using specific coefficients
a_test, b_test, c_test = 1.0, 2.0, 1.0  # Example coefficients
true_derivative = 2 * a_test * test_value + b_test
print(f"True derivative at x = 2.0: {true_derivative[0][0]}")