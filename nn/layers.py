import numpy as np

class Layer:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, grad):
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        pass

# Activations functions
# So far we have
# [sigmoid, relu, tanh]

def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return np.tanh(x)


