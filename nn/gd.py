# Gradient Descent
from network import NeuralNetwork
import numpy as np

def fit(network: NeuralNetwork, X_train: np.ndarray, y_train: np.ndarray, epochs: int, learning_rate: float):
    """
    Trains a neural network using gradient descent.

    Args:
        network (NeuralNetwork): The neural network to train.
        X_train (np.ndarray): The training input data.
        y_train (np.ndarray): The training output data.
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate for gradient descent.
    """
    for epoch in range(epochs):
        error = 0
        for x, y in zip(X_train, y_train):
            # Forward pass
            output = network.forward(x)

            # Calculate error
            error += network.loss(y, output)

            # Backward pass
            network.backward(y, learning_rate)

        # Calculate average error for the epoch
        error /= len(X_train)
        print(f"Epoch {epoch + 1}/{epochs}, Error: {error:.4f}")
