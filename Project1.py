import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

nNums = 1000
train_data = np.random.randint(-10,10,nNums)
train_labels = []
test_labels  = []

test_data = []
test_data.append(int(input('Type one integrer between -10 and 10 seperated by a space: ')))
test_data.append(int(input('Type one integrer between -10 and 10 seperated by a space: ')))

train_data   = train_data.reshape(500,2)

train_labels = np.sum(train_data,axis=1)
test_labels  = np.sum(test_data)

# convert to tensor
train_dataT   = torch.tensor( train_data ).float()
train_labelsT = torch.tensor(train_labels, dtype=torch.float32).reshape(-1, 1)

test_dataT = torch.tensor(test_data, dtype=torch.float32).reshape(1, 2)
test_labelsT = torch.tensor([[test_labels]], dtype=torch.float32)


# convert into PyTorch Datasets
train_data = torch.utils.data.TensorDataset(train_dataT,train_labelsT)
test_data  = torch.utils.data.TensorDataset(test_dataT,test_labelsT)

# translate into dataloader objects
batchsize    = 2
train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True)
test_loader  = DataLoader(test_data,batch_size=2)

X, y = next(iter(train_loader))
print(X.shape, y.shape)

X, y = next(iter(test_loader))
print(X.shape, y.shape)


def createTheModel():

    class calcnet(nn.Module):
        def __init__(self):
            super().__init__()

            self.input  = nn.Linear(2,16)
            self.fc1    = nn.Linear(16,4)
            self.output = nn.Linear(4,1)

        def forward(self,x):

            x = self.input(x)
            x = self.fc1(x)

            return self.output(x)

    net = calcnet()
    lossfun = nn.MSELoss()
    optim = torch.optim.Adam(net.parameters(),lr=.001)

    return net,lossfun,optim


def trainThemodel():

    net,lossfun,optim = createTheModel()

    numepochs = 1000

    trainAcc = []
    testAcc  = []
    losses = torch.zeros(numepochs)

    for epochi in range(numepochs):

        net.train()

        batchAcc  = []
        batchLoss = []
        for X,y in train_loader:

            yHat = net(X)
            loss = lossfun(yHat,y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            batchAcc.append(torch.mean(torch.abs(yHat - y)).item())
            batchLoss.append( loss.item() )

        trainAcc.append( np.mean(batchAcc) )
        losses[epochi] = np.mean( batchLoss )

        net.eval()

        X,y  = next(iter(test_loader))
        with torch.no_grad():
            yHat = net(X)

        testAcc.append(torch.mean(torch.abs(yHat - y)).item())

    return yHat

yHat = trainThemodel()

print(f'The sum is:{yHat}')