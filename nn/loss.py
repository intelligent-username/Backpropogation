# Loss functions

import numpy as np

# Only Mean Squared Error right now, might implement more later but probably unnecessary for this generic demo

def MSE(y_true, y_pred):
    """
    Calculates the Mean Squared Error loss.

    Args:
        y_true: The true target values.
        y_pred: The predicted values.

    Returns:
        The mean squared error.
    """
    # Divide by two to make the constant for the derivative simpler
    return (((y_true - y_pred) ** 2).mean()/2)


def MSE_derivative(y_true, y_pred):
    """
    Calculates the derivative of the Mean Squared Error loss.

    Args:
        y_true: The true target values.
        y_pred: The predicted values.

    Returns:
        The derivative of the MSE loss, averaged.
    """
    return (y_pred - y_true) / y_true.size


def huber_loss(y_true, y_pred, delta=1.0):
    """
    Calculates the Huber loss between true and predicted values.

    Args:
        y_true: The true target values.
        y_pred: The predicted values.
        delta (float): The point where the loss changes from quadratic to linear.

    Returns:
        The mean Huber loss.
    """
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = np.square(error) / 2
    linear_loss = delta * (np.abs(error) - delta / 2)
    return np.where(is_small_error, squared_loss, linear_loss).mean()

def huber_loss_derivative(y_true, y_pred, delta=1.0):
    """
    Calculates the derivative of the Huber loss.

    Args:
        y_true: The true target values.
        y_pred: The predicted values.
        delta (float): The threshold for the derivative.

    Returns:
        The derivative of the Huber loss, averaged.
    """
    error = y_pred - y_true
    derivative = np.where(np.abs(error) <= delta, error, delta * np.sign(error))
    return derivative / y_true.size

def mae(y_true, y_pred):
    """
    Calculates the Mean Absolute Error loss.

    Args:
        y_true: The true target values.
        y_pred: The predicted values.

    Returns:
        The mean absolute error.
    """
    return np.abs(y_true - y_pred).mean()

def mae_derivative(y_true, y_pred):
    """
    Calculates the derivative of MAE at y_pred.

    Args:
        y_true: The true target values.
        y_pred: The predicted values.

    Returns:
        The derivative of the MAE loss, averaged.
    """
    return np.sign(y_pred - y_true) / y_true.size
