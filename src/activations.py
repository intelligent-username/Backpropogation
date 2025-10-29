# Activations functions
# So far we have
# [sigmoid, relu, tanh]

import numpy as np

def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return np.tanh(x)
