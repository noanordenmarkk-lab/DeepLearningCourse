import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo 
import time
  
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

data = pd.read_csv(url,sep=';')

data = data[data['total sulfur dioxide']<200]

### z-score all variables except for quality

# find the columns we want to normalize (all except quality)
cols2zscore = data.keys()
cols2zscore = cols2zscore.drop('quality')

data[cols2zscore] = data[cols2zscore].apply(stats.zscore)

# create a new column for binarized (boolean) quality
data['boolQuality'] = 0
data.loc[data['quality'] > 5, 'boolQuality'] = 1

# convert from pandas dataframe to tensor
dataT  = torch.tensor( data[cols2zscore].values ).float()
labels = torch.tensor( data['boolQuality'].values ).float()

# we'll actually need the labels to be a "tensor"
labels = labels[:,None]

# use scikitlearn to split the data
train_data,test_data, train_labels,test_labels = train_test_split(dataT, labels, test_size=.1)

# then convert them into PyTorch Datasets (note: already converted to tensors)
train_data = TensorDataset(train_data,train_labels)
test_data  = TensorDataset(test_data,test_labels)


# finally, translate into dataloader objects
batchsize    = 64
train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True)
test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

class WineModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(11,64)
        self.hidden1 = nn.Linear(64,64)
        self.hidden2 = nn.Linear(64,64)
        self.output = nn.Linear(64,1)

    def forward(self,x):

       x = F.relu(self.input(x)) 
       x = F.relu(self.hidden1(x))
       x = F.relu(self.hidden2(x)) 

       return self.output(x)


#training
ANNWine = WineModel()

learning_rate = 0.001
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(ANNWine.parameters(),lr=learning_rate)

numepochs = 1000

trainAcc = []
testAcc = []
losses = torch.zeros(numepochs)

for epochi in range(numepochs):

    ANNWine.train()

    batchAcc = []
    batchLoss = []

    for X,y in train_loader:

      yHat = ANNWine(X)
      loss = loss_fn(yHat,y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      batchAcc.append(100*torch.mean(((yHat > 0).float() == y).float()).item())
      batchLoss.append( loss.item() )

    trainAcc.append(np.mean(batchAcc))
    losses[epochi] = np.mean(batchLoss)

    ANNWine.eval()
    X,y = next(iter(test_loader))
    with torch.no_grad():
       yHat = ANNWine(X)
    testAcc.append(100 * torch.mean(((yHat > 0) == y).float()).item())

