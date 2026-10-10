import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

nNums = 1000
data = np.random.randint(-10,10,nNums)

test_data = []
test_data.append(int(input('Type one integrer between -10 and 10 seperated by a space: ')))
test_data.append(int(input('Type one integrer between -10 and 10 seperated by a space: ')))

# Step 4: translate into dataloader objects
batchsize    = 2
train_loader = DataLoader(data,batch_size=batchsize,shuffle=True)
test_loader  = DataLoader(test_data,batch_size=2)



