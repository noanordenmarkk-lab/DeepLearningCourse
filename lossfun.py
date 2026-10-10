import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np

import matplotlib.pyplot as plt
from IPython import display

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

def CreateANN():
    class ANN(nn.Module):
        def __init__(self):
            super().__init__()

            self.input = nn.Linear(2,8)

            self.fc1 = nn.Linear(8,8)

            self.output = nn.Linear(8,3)

        def forward(self,x):
            x = F.relu(self.input(x))
            x = F.relu(self.fc1(x))
            x = self.output(x)
            return x

    net = ANN()
    lossfun = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(net.parameters(),lr=0.01)

    return net,lossfun,opt

def trainTheModel(net,lossfun,opt):
    nEpochs  = 1000

    losses   = torch.zeros(nEpochs)
    trainAcc = []
    testAcc  = []

    for epochi in range(nEpochs):

        net.train() #turn on dropout

        batchLoss = []
        batchAcc  = []
        for X,y in train_loader:

            yHat = net(X)
            loss = lossfun(yHat,y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            batchLoss.append(loss.item())
            #xompute accuracy
            matches        = torch.argmax(yHat,axis=1) == y #booleans
            matchesNumeric = matches.float()                #conmvert to 0/1
            accuracyPct    = 100*torch.mean(matchesNumeric) #avarage and x100
            batchAcc.append(accuracyPct.item())                    #add to list
            #end of batch loop
        
        trainAcc.append( np.mean(batchAcc) )

        losses[epochi] = np.mean(batchLoss)

        net.eval() #turn off dropout
        X,y = next(iter(test_loader))
        with torch.no_grad():
            yHat = net(X)

        testAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1) == y).float()).item() )
        #end of epoch loop

    return losses,trainAcc,testAcc,net

net,lossfun,opt = CreateANN()
losses,trainAcc,testAcc,net = trainTheModel(net,lossfun,opt)

fig,ax = plt.subplots(1,2,figsize=(13,4))

ax[0].plot([loss.item() for loss in losses])
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_title('Losses')

ax[1].plot(trainAcc,label='Train')
ax[1].plot(testAcc,label='Test')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_xlabel('Epochs')
ax[1].set_title('Model Performance')
ax[1].legend()

plt.show()