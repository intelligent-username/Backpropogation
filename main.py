from nn.network import NeuralNetwork
from nn.layer import Layer
from nn.loss import cross_entropy, cross_entropy_derivative # Best for classification yo

from nn.activations import relu, relu_derivative, softmax, softmax_derivative_passthrough

from nn.gd import fit
from utils.loader import load_digits_dataset
from utils.splitter import data_split

import numpy as np

def main():
    print("Yo. We're going to build and train this neural network right now.")

    np.random.seed(42)

    nn = NeuralNetwork(loss=cross_entropy, loss_derivative=cross_entropy_derivative)

    # 16x16 inputs = 256 features
    layer1 = Layer(input_size=256, output_size=128, activation=relu, d_activation=relu_derivative, mask=None)
    layer2 = Layer(input_size=128, output_size=64, activation=relu, d_activation=relu_derivative, mask=None)
    # Use softmax on output; derivative passthrough so CE gradient isn’t multiplied again
    layer3 = Layer(input_size=64, output_size=10, activation=softmax, d_activation=softmax_derivative_passthrough, mask=None)

    nn.add_layers([layer1, layer2, layer3])

    print("Now we're loading the dataset...")

    X, Y = load_digits_dataset()  # normalized with scale255

    X_train, Y_train, X_val, Y_val, X_test, Y_test = data_split(X, Y, train_ratio=0.8, val_ratio=0.1, seed=42)

    print("Training...")
    test_error = fit(nn, X_train, Y_train, X_val, Y_val, X_test, Y_test, max_epochs=200, learning_rate=0.005)

    nn.save_model("digits_nn_model.pkl")

    print("Done training.")
    print(f"Final Test Error: {test_error:.4f}")

    print("Now let's actually put this to use.")

    # Re-test on the actual training data itself to check for overfitting
    train_correct = 0
    train_total = len(X_train)
    for i in range(train_total):
        output = nn.forward(X_train[i])
        predicted_label = np.argmax(output, axis=1)[0]
        actual_label = np.argmax(Y_train[i])

        if train_total % (i + 1) == 0:
            print(f"Train Image {i + 1}: Predicted Label = {predicted_label}, Actual Label = {actual_label}")

        if predicted_label == actual_label:
            train_correct += 1

    train_accuracy = (train_correct / train_total * 100) if train_total > 0 else 0.0
    print(f"Train Accuracy: {train_accuracy:.2f}%")

    # Real test set
    X, Y = load_digits_dataset("data/digits/tests", target_size=(16,16))
    correct = 0
    total = len(X)
    for i in range(total):
        output = nn.forward(X[i])
        predicted_label = np.argmax(output, axis=1)[0]
        actual_label = np.argmax(Y[i])
        if train_total % (i+1) == 0:
            print(f"Test Image {i + 1}: Predicted Label = {predicted_label}, Actual Label = {actual_label}")

        if predicted_label == actual_label:
            correct += 1

    accuracy = (correct / total * 100) if total > 0 else 0.0 # Avoid division by zero although it should never happen
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("See, this is why we need convolutional neural networks, since Dense networks expect flat 1D inputs.")

if __name__ == "__main__":
    main()
