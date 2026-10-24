import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scipy.stats as stats

data = np.loadtxt(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    delimiter=",",
    dtype=str
)

mask = ~(data == '?').any(axis=1)

data = data[mask]

labels = data[:,-1].astype(float)
data   = data[:,:-1].astype(float)

dataf = pd.DataFrame(data)

print(dataf.shape)
print( dataf.head() )