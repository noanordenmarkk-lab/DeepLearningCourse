import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

import seaborn as sns
iris   = sns.load_dataset('iris')

data   = torch.tensor( iris[iris.columns[0:4]].values ).float()

labels = torch.zeros(len(data), dtype=torch.long)
labels[iris.species=='versicolor']   = 1
labels[iris.species=='virginica']    = 2

train_data,test_data, train_labels,test_labels = \
train_test_split(data,labels, test_size=.02)

train_data = TensorDataset(train_data,train_labels)
test_data  = TensorDataset(test_data,test_labels)

batchsize    = 16
train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)
test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

def createANewModel():

    ANNiris = nn.Sequential(
        nn.Linear(4,64),
        nn.ReLU(),
        nn.Linear(64,64),
        nn.ReLU(),
        nn.Linear(64,3),
        )

    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ANNiris.parameters(),lr=0.0005)
    return ANNiris, lossfun, optimizer

numepochs = 500

def trainTheModel():

    ANNiris, lossfun, optimizer = createANewModel()

    trainAcc = []
    testAcc  = []
    losses   = []

    for epochi in range(numepochs):

        batchAcc  = []
        batchLoss = []
        for X,y in train_loader:

            yHat = ANNiris(X)
            loss = lossfun(yHat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1) == y).float()).item() )
            batchLoss.append( loss.item() )

        trainAcc.append( np.mean(batchAcc) )
        losses.append( np.mean(batchLoss) )

    return trainAcc, losses, testAcc
