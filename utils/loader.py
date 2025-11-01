# Load in data and convert to some 'standard format' (i.e. clean, normalize, split into sets, etc. etc.)

import numpy as np
import pandas as pd
import csv
from PIL import Image
import os

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

def normalize_data(data, range_min=0, range_max=1):
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min) * (range_max - range_min) + range_min


def load_digits_dataset(path: str = 'data/digits', target_size: tuple = (16, 16)) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the digits dataset from the specified path.

    Args:
        path (str): The path to the digits dataset directory.
        target_size (tuple): The target size to resize the images to.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the images and labels.
    """
    images = []
    labels = []
    num_classes = 10

    for label in range(num_classes):
        dir_path = os.path.join(path, str(label))
        if not os.path.isdir(dir_path):
            continue
        
        for filename in os.listdir(dir_path):
            try:
                img_path = os.path.join(dir_path, filename)
                with Image.open(img_path) as img:
                    # Convert to grayscale, resize, and convert to numpy array
                    img = img.convert('L').resize(target_size, Image.Resampling.LANCZOS)
                    img_array = np.array(img)
                    
                    # Flatten the image and append
                    images.append(img_array.flatten())
                    labels.append(label)
            except Exception as e:
                print(f"Could not process file {filename}: {e}")

    # Convert lists to numpy arrays
    X = np.array(images)
    Y = np.array(labels)

    # Normalize image data and one-hot encode labels
    X = normalize_data(X)
    Y = one_hot_encode(Y, num_classes)

    return X, Y
