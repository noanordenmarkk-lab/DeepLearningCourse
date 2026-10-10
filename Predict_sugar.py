### import libraries

# for DL modeling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

# for number-crunching
import numpy as np
import scipy.stats as stats

# for dataset management
import pandas as pd

# for timing computations
import time

# for data visualization
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# import the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url,sep=';')
data = data[data['total sulfur dioxide']<200] # drop a few outliers


cols2 = data.keys() #gives you all the columnnames in the data
cols2 = cols2.drop('residual sugar') #takes away quality from the data

# convert from pandas dataframe to tensor
dataT  = torch.tensor( data[cols2].values ).float()
labels = torch.tensor( data['residual sugar'].values ).float()
labels = labels[:,None] # transform to matrix

# use scikitlearn to split the data
train_data,test_data, train_labels,test_labels = train_test_split(dataT, labels, test_size=.1)

#Min max scaling

mins = train_data.amin(dim=0)
maxs = train_data.amax(dim=0)
train_data = (train_data - mins) / (maxs-mins)
test_data  = (test_data - mins)  / (maxs-mins)

#labels
lmin = train_labels.amin(dim=0)
lmax = train_labels.amax(dim=0)
train_labels = (train_labels - lmin) / (lmax - lmin)
test_labels  = (test_labels  - lmin) / (lmax - lmin)

# then convert them into PyTorch Datasets (note: already converted to tensors)
train_dataDataset = TensorDataset(train_data,train_labels)
test_dataDataset  = TensorDataset(test_data,test_labels)

#convert to loaders
train_loader = DataLoader(train_dataDataset, shuffle=True, batch_size=32, drop_last=True)
test_loader  = DataLoader(test_dataDataset, batch_size=test_dataDataset.tensors[0].shape[0])

#create a class for the model
class ANNwine(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(11,32)

        #hidden layers
        self.fc1 = nn.Linear(32,64)
        self.fc2 = nn.Linear(64,32)

        #output layer
        self.output = nn.Linear(32,1)

    def forward(self,x):
         x = F.relu( self.input(x) )
         x = F.relu( self.fc1(x) ) 
         x = F.relu( self.fc2(x) )

         return self.output(x)

numepochs = 600

def trainTheModel(winenet):

    lossfun = nn.MSELoss()
    optimizer = torch.optim.Adam(winenet.parameters(),lr=0.001)

    testMAE  = []
    trainMAE = []
    Losses   = torch.zeros(numepochs)

    for epochi in range(numepochs):

        winenet.train()

        batchMAE  = []
        batchLoss = []
        for X,y in train_loader:

            yHat = winenet(X)
            loss = lossfun(yHat,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchMAE.append(torch.mean(torch.abs(yHat - y)).item())
            batchLoss.append( loss.item())
            #end of batch loop

        trainMAE.append( np.mean(batchMAE) )
        Losses[epochi] = np.mean(batchLoss)

        #testMAE
        winenet.eval()
        X,y = next(iter(test_loader))
        with torch.no_grad():
            yHat = winenet(X)
        testMAE.append(torch.mean(torch.abs(yHat - y)).item())

    return trainMAE,testMAE,Losses

winenet = ANNwine()
trainMAE,testMAE,Losses = trainTheModel(winenet)
fig,ax = plt.subplots(1,3,figsize=(17,7))

ax[0].plot(trainMAE)
ax[0].set_title('Train MAE')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('MAE')
ax[0].grid()

ax[1].plot(testMAE)
ax[1].set_title('Test MAE')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('MAE')
ax[1].grid()

ax[2].plot(Losses)
ax[2].set_title('Losses')
ax[2].set_xlabel('Epoch')
ax[2].set_ylabel('Loss')
ax[2].grid()

plt.show()