import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np

import matplotlib.pyplot as plt

nPerClust = 300
blur = 1

A = [ 1, 1 ]
B = [ 5, 1 ]
C = [ 4, 4 ]

a = [ A[0]+np.random.randn(nPerClust)*blur , A[1]+np.random.randn(nPerClust)*blur]
b = [ B[0]+np.random.randn(nPerClust)*blur , B[1]+np.random.randn(nPerClust)*blur]
c = [ C[0]+np.random.randn(nPerClust)*blur , C[1]+np.random.randn(nPerClust)*blur]

labels_np = np.hstack(( np.zeros((nPerClust)),
                        np.ones( (nPerClust)),
                      1+np.ones( (nPerClust)) ))

data_np = np.hstack((a,b,c)).T

data   = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).long()

#use sickitlearn to split the data
train_data,test_data, train_labels,test_labels = train_test_split(data,labels,test_size=0.2)

#then convert them into pytorch datasets
train_data = TensorDataset(train_data,train_labels)
test_data  = TensorDataset(test_data,test_labels)

#finally, translate into dataloaders
batch_size   = 16
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0],shuffle=False)

def createANN(momentum):

    class ANN(nn.Module):
        def __init__(self):
            super().__init__()

            self.input = nn.Linear(2,8)
            self.fc1   = nn.Linear(8,8)
            self.output = nn.Linear(8,3)

        def forward(self,x):
            x = F.relu(self.input(x))
            x = F.relu(self.fc1(x))
            return self.output(x)

    net = ANN()
    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=0.001,momentum=momentum)

    return net,lossfun,optimizer

numepochs = 50

def trainModel(momentum):

    net,lossfun,optimizer = createANN(momentum)

    losses   = torch.zeros(numepochs)
    trainAcc = []
    testAcc  = []

    for epochi in range(numepochs):

        batchAcc  = []
        batchLoss = []
        for X,y in train_loader:

            yHat = net(X)
            loss = lossfun(yHat,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchLoss.append(loss.item())

            matches = torch.argmax(yHat,axis=1) == y
            matchesNumeric = matches.float()
            accuracyPct = 100*torch.mean(matchesNumeric)
            batchAcc.append( accuracyPct )

        trainAcc.append( np.mean(batchAcc) )

        losses[epochi] = np.mean(batchLoss)

        net.eval()
        X,y = next(iter(test_loader))
        with torch.no_grad():
            yHat = net(X)

        testAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1)==y).float()))

    return trainAcc,testAcc,losses,net

momenta = [0,.5,.9,.95,.999]

results = np.zeros((numepochs, len(momenta), 3))

for idx,mom in enumerate(momenta):
    trainAcc,testAcc,losses,net = trainModel(mom)
    results[:,idx,0] = losses
    results[:,idx,1] = trainAcc
    results[:,idx,2] = testAcc

fig,ax = plt.subplots(1,3,figsize=(16,5))

for i in range(3):
    ax[i].plot(results[:,:,i])
    ax[i].legend(momenta)
    ax[i].set_xlabel('Epochs')
    if i==0:
        ax[i].set_ylabel('Loss')
    else:
        ax[i].set_ylabel('Accuracy (%)')

ax[0].set_title('Losses')
ax[0].set_title('Train')
ax[0].set_title('Test')

plt.show()