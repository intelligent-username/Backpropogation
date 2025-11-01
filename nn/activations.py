# Activation Functions
# So far [ReLU, Sigmoid, Tanh]

import numpy as np

def relu(x):
    """
    ReLU: Rectified Linear Unit
    Is 0 when x < 0, else x
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of ReLU
    Not defined at 0, just pick 0.
    """
    return (x > 0).astype(float)

def sigmoid(x):
    """
    The Sigmoid Activation Function
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivative of the Sigmoid Activation Function
    """
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    """
    Derivative of the Tanh Activation Function
    """
    return 1 - (np.tanh(x) ** 2)

def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax activation function.
    """
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-12)

# NOTE: dL/dz = (y_pred - y_true)
# That's why we "don't" multiply by the derivative again
def softmax_derivative_passthrough(z: np.ndarray) -> np.ndarray:
    """
    Softmax derivative
    """
    return np.ones_like(z)
