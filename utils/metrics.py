import numpy as np

# Measure how accurate the model is

# Return accuracy, precision, specificity.

# Could use more, take inspiration from https://github.com/intelligent-username/Similarity-Metrics

def accuracy(model, data):
    """
    Accuracy: (True Positives + True Negatives) / Total Samples
    """
    return np.mean(model.predict(data.X) == data.y)

def precision(model, data):
    """
    Precision: (True Positives) / (True Positives + False Positives)
    """
    preds = model.predict(data.X)
    true_positives = np.sum(preds == data.y)
    false_positives = np.sum(preds != data.y)
    return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

def specificity(model, data):
    """
    Specificity: (True Negatives) / (True Negatives + False Positives)
    """
    preds = model.predict(data.X)
    true_negatives = np.sum((preds != data.y) & (data.y == 0))
    false_positives = np.sum((preds != data.y) & (data.y == 1))
    return true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0