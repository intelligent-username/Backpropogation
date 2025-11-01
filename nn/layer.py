# Individual layers

import numpy as np

from typing import Callable

class Layer:
    weights: np.ndarray         # How much each neuron weighs each of its inputs
        # Note: each row corresponds to the neuron (not explicitly labeled btw)
        # And each column corresponds to the weight that neuron uses for the corresponding input
        # Shape: (input_size, output_size)


    mask: np.ndarray
        # Mask: in case some neurons aren't connected to all other neurons in the previous layer.
        # Same shape as weights.
        # Can also be used for Dropout
        # 0: weight is not used (at all) else is used

    
    biases: np.ndarray          # One bias ('vertical translation') for each output neuron
    activation: Callable        # Activation function
    d_activation: Callable      # Derivative of activation
    input_cache: np.ndarray     # Stores input X for backprop
    z_cache: np.ndarray         # Stores weighted sum before activation


    def __init__(self, input_size: int, output_size: int, 
                 activation: Callable, d_activation: Callable, 
                 mask: np.ndarray | None = None):

        # Initial weights (random)
        # Used to be zero but that'll probably break the initial few steps at least

        self.weights = np.random.randn(input_size, output_size) * 0.01 # Initialize randomly
        self.biases = np.zeros((1, output_size))
        self.mask = np.ones((input_size, output_size)) if mask is None else mask

        # Activation functions
        self.activation = activation
        self.d_activation = d_activation

        # Caches for forward/backward
        self.input_cache = None
        self.z_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        The forward pass of the current layer.
        Should iterate until the output is reached. Then reuse for the backward propagation and weight adjustment.
        """

        # Store input
        self.input_cache = x

        # Apply mask to weights ( vectorized btw :) ) 
        cur_weights = self.weights * self.mask

        # Compute weighted sum
        z = np.dot(x, cur_weights) + self.biases
        self.z_cache = z
        # z is the conventional name

        # Pass it through the activation
        a = self.activation(z)
        return a
        # Note: we are returning the 'vector of inputs' for the next layer.

    def backward(self, grad_output: np.ndarray, lr: float) -> np.ndarray:
        # grad_output: derivative of loss w.r.t. output of this layer
        # lr: learning rate
        # May be adjusted through gradient descent algorithm (or its optimizations)

        # Derivative of activation
        dz = grad_output * self.d_activation(self.z_cache)

        # Gradients
        grad_w = np.dot(self.input_cache.T, dz) * self.mask
        grad_b = np.sum(dz, axis=0, keepdims=True)
        grad_input = np.dot(dz, (self.weights * self.mask).T)

        # Update parameters
        self.weights -= lr * grad_w
        self.biases -= lr * grad_b

        return grad_input
