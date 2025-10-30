import numpy as np
import pickle
from typing import Callable


class NeuralNetwork:
    def __init__(self, loss: Callable, loss_derivative: Callable):
        self.layers = []
        self.forward_val = None
        self.loss = loss
        self.loss_derivative = loss_derivative

    def add_layer(self, layer) -> None:
        self.layers.append(layer)

    def forward(self, x) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        self.forward_val = x
        return x

    def backward(self, y_true, learning_rate) -> None:
        # Compute initial gradient from loss derivative
        grad = self.loss_derivative(y_true, self.forward_val)
        
        # Backpropagate through the rest of the layers in reverse order
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def save_model(self, path: str) -> None:
        """
        Saves the neural network model to the specified path using pickle.

        Args:
            path (str): The file path to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(path: str) -> 'NeuralNetwork':
        """
        Loads a neural network model from the specified path using pickle.

        Args:
            path (str): The file path to load the model from.

        Returns:
            NeuralNetwork: The loaded neural network model.
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
