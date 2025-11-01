# Make train, validation, and test splits

import numpy as np
from typing import Tuple

def data_split(X : np.ndarray, Y: np.ndarray, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        X (np.ndarray): The input data.
        Y (np.ndarray): The target data.
        train_ratio (float): The ratio of the training set.
        val_ratio (float): The ratio of the validation set.

    Returns:
        tuple: (X_train, Y_train, X_val, Y_val, X_test, Y_test).
            a tuple of numpy arrays representing the training, validation, and test sets respectively.
    """

    n = X.shape[0]

    indices = np.arange(n)

    np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]

    # Split ratios
    n = X.shape[0]
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    # Partition
    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

