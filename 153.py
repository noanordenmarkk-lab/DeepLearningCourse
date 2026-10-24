import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import time

import matplotlib.pyplot as plt

# import dataset
data = np.loadtxt(
    open(r'C:\Users\Administrator\Documents\Programming\DeepLearningCourse\Datasets\mnist_train_small.csv', 'rb'),
    delimiter=','
)

#proccess data
data = data[:,1:]
dataNorm = data / np.max(data)
dataT = torch.tensor( dataNorm ).float()

dataT_loader = DataLoader(dataT, batch_size=32, shuffle=True, drop_last=True)

def createTheMNISTAE(ec_units, bn_units):

    class aenet(nn.Module):
        def __init__(self, ec_units, bn_units):
            super().__init__()

            self.input  = nn.Linear(784,ec_units)
            self.enc    = nn.Linear(ec_units,bn_units)
            self.lat    = nn.Linear(bn_units,ec_units)
            self.dec = nn.Linear(ec_units,784)

        def forward(self,x):
            x = F.relu(self.input(x))
            x = F.relu(self.enc(x))
            x = F.relu(self.lat(x))
            
            y = torch.sigmoid(self.dec(x))
            return y

    net     = aenet(ec_units, bn_units)
    lossfun = nn.MSELoss()
    opt     = torch.optim.Adam(net.parameters(), lr=.001)

    return net,lossfun,opt

def trainTheModel(ec_units, bn_units):

    numepochs = 20

    net,lossfun,opt = createTheMNISTAE(ec_units, bn_units)

    losses = torch.zeros(numepochs)

    for epochi in range(numepochs):

        batchLoss = []
        for X in dataT_loader:

            xHat = net(X)
            loss = lossfun(xHat,X)

            opt.zero_grad()
            loss.backward()
            opt.step()

            batchLoss.append( loss.item() )

        losses[epochi] = np.mean(batchLoss)

    return losses,net

def CodeChallange():
    ec_units = np.linspace(10, 500, 12).astype(int)
    bn_units = np.linspace(5, 100, 8).astype(int)

    finallosses = []
    completed   = 0
    for i in range(len(ec_units)):
        for i2 in range(len(bn_units)):
            losses,net = trainTheModel(ec_units[i], bn_units[i2])
            finallosses.append(torch.mean(losses[-3:]).item())

            completed +=1
            print(f'\rProgress: {(completed/96)*100}%', end='', flush=True)
            time.sleep(0.05)

    fig,ax = plt.subplots(1,1,figsize=(17,7))
    ax.set_ylabel('number of encoder/decoder units')
    ax.set_xlabel('numver of bottleneck units')

    ax.set_xticks(range(8), bn_units)
    ax.set_yticks(range(12), ec_units)

    finallosses = np.array(finallosses).reshape(12, 8)

    ax.imshow(finallosses, cmap='Purples', origin='lower', aspect='auto')
    ax.colorbar(label='Loss')

    plt.show()

CodeChallange()