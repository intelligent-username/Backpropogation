import numpy as np
from datetime import datetime
import pickle
from typing import Callable
import os


class NeuralNetwork:
    def __init__(self, loss: Callable, loss_derivative: Callable):
        self.layers = []
        self.forward_val = None
        self.loss = loss
        self.loss_derivative = loss_derivative

    def add_layer(self, layer) -> None:
        self.layers.append(layer)
    
    def add_layers(self, layers: list) -> None:
        self.layers.extend(layers)

    def forward(self, x) -> np.ndarray:
        # ensure (batch, features)
        if isinstance(x, np.ndarray) and x.ndim == 1:
            x = x.reshape(1, -1)
        for layer in self.layers:
            x = layer.forward(x)
        self.forward_val = x        # Careful not to accidentally store/access an old value in the future
        return self.forward_val

    def backward(self, y_true, learning_rate) -> None:

        if isinstance(y_true, np.ndarray) and y_true.ndim == 1:
            y_true = y_true.reshape(1, -1)

        grad = self.loss_derivative(y_true, self.forward_val)
        # Backpropagate through the rest of the layers in reverse order
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def save_model(self, path: str="") -> None:
        """
        Saves the neural network model to the specified path using pickle.

        Args:
            path (str): A file name, a directory, or empty.
                        - "": saves to models/nn_model_<timestamp>.pkl
                        - "name.pkl": saves to models/name.pkl
                        - "dir/": saves to dir/nn_model_<timestamp>.pkl
                        - "dir/name" (no .pkl): saves to dir/name_<timestamp>.pkl
                        - "dir/name.pkl": saves to dir/name.pkl
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not path:
            dirpath = "models"
            filename = f"nn_model_{timestamp}.pkl"
        else:
            # If path ends with a separator, treat as directory only
            if path.endswith(("/", "\\")):
                dirpath = path.rstrip("/\\")
                filename = f"nn_model_{timestamp}.pkl"
            else:
                dirpath = os.path.dirname(path) or "models"
                base = os.path.basename(path)
                root, ext = os.path.splitext(base)
                if ext.lower() == ".pkl":
                    filename = base
                else:
                    filename = f"{root}_{timestamp}.pkl"

        os.makedirs(dirpath, exist_ok=True)
        full_path = os.path.join(dirpath, filename)

        with open(full_path, 'wb') as f:
            pickle.dump(self, f)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Uses the neural network to label the input data.
        """
        return self.forward(x)
    
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
    
    def __str__(self):
        """
        'Prints' the neural network by creating a matplotlib visualization
        """
