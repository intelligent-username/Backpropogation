# Load in data and convert to some 'standard format' (i.e. clean, normalize, split into sets, etc. etc.)

import numpy as np
import pandas as pd
import csv

def one_hot_encode(labels, num_classes):
    pass

def normalize_data(data, range_min=0, range_max=1):
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min) * (range_max - range_min) + range_min


