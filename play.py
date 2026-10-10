# import libraries
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# NEW!
import sklearn.metrics as skm

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# import dataset (comes with colab!)
data = np.loadtxt(
    open(r'C:\Users\Administrator\Documents\Programming\DeepLearningCourse\Datasets\mnist_train_small.csv', 'rb'),
    delimiter=','
)

#take away sevens so you only have 500 left
bissevens = data[:,0] == 7
sevens    = data[bissevens] 
others    = data[~bissevens]

data = np.concatenate((others,sevens[:500]), axis=0)


# extract labels (number IDs) and remove from data
labels = data[:,0]
data   = data[:,1:]

# normalize the data to a range of [0 1]
dataNorm = data / np.max(data)

# Step 1: convert to tensor
dataT   = torch.tensor( dataNorm ).float()
labelsT = torch.tensor( labels ).long()

# Step 2: use scikitlearn to split the data
train_data,test_data, train_labels,test_labels = train_test_split(dataT, labelsT, test_size=.1)

# Step 3: convert into PyTorch Datasets
train_data = torch.utils.data.TensorDataset(train_data,train_labels)
test_data  = torch.utils.data.TensorDataset(test_data,test_labels)

# Step 4: translate into dataloader objects
batchsize    = 32
train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)
test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

fig, ax = plt.subplots(1, 1, figsize=(17, 7))

counts = [np.sum(labels == i) for i in range(10)]

ax.bar(range(10), counts)
ax.set_ylabel('N occurrences')
ax.set_xlabel('Digits')
ax.set_xticks(range(10))

plt.show()