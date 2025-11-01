from nn.network import NeuralNetwork
from nn.layer import Layer
from nn.loss import mse, mse_derivative
from nn.activations import sigmoid, sigmoid_derivative, tanh, tanh_derivative
from nn.gd import fit           # Could implement Adam, RMSProp, etc. etc. but gradient descent is sufficient rn. Can make a massive library later.
from utils.loader import load_digits_dataset
from utils.splitter import data_split

import numpy as np

def main():

    print("Yo. We're going to build and train this neural network right now.")

    nn = NeuralNetwork(loss=mse, loss_derivative=mse_derivative)

    # For example, let's do 2 layers (and input-output layers of course)

    print("Our goal is to make a mini vision engine that can recognize images of digits as digits. For now, we're only testing single digits, but in the future we could implement a second layer that splits an image into its individual digits for example, runs it through this network to get each individual digits, then combines them to write out the full number. In that case, we could add decimal points, signs, scientific notatoin, and the like. But for now, I'm going to keep it basic.")
    
    # This one takes in the inputs
    # Regularized everything to be 

    layer1 = Layer(input_size = 256, output_size = 16, activation = tanh, d_activation = tanh_derivative, mask = None)
    # Note: images will be scaled to 16x16
    
    layer2 = Layer(input_size = 16, output_size = 8, activation = sigmoid, d_activation = sigmoid_derivative, mask = None)
    layer3 = Layer(input_size = 8, output_size = 10, activation = sigmoid, d_activation = sigmoid_derivative, mask = None)

    nn.add_layers([layer1, layer2, layer3])

    print("Neural network has been created. Now time to regularize the data before training on it.")

    print("Unfortunately, we compressed everything down to be 8x8 in order to regularize. Lots of information is lost.")

    X, Y = load_digits_dataset()

    X_train, Y_train, X_val, Y_val, X_test, Y_test = data_split(X, Y, train_ratio=0.8, val_ratio=0.1)

    # Now just train
    print("Training...")

    test_error = fit(nn, X_train, Y_train, X_val, Y_val, X_test, Y_test, max_epochs=100, learning_rate=0.01)

    nn.save_model("digits_nn_model.pkl")

    print("Done training.")
    print(f"Final Test Error: {test_error:.4f}")

    print("Now let's actually put this to use.")


if __name__ == "__main__":
    main()
