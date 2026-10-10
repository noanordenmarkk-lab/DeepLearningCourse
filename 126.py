import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import copy

nPerClust = 300
blur = 1

A = [ 1, 1 ]
B = [ 5, 1 ]
C = [ 4, 3 ]

a = [ A[0]+np.random.randn(nPerClust)*blur , A[1]+np.random.randn(nPerClust)*blur]
b = [ B[0]+np.random.randn(nPerClust)*blur , B[1]+np.random.randn(nPerClust)*blur]
c = [ C[0]+np.random.randn(nPerClust)*blur , C[1]+np.random.randn(nPerClust)*blur]

labels_np = np.hstack(( np.zeros((nPerClust)),
                        np.ones( (nPerClust)),
                      1+np.ones( (nPerClust)) ))

data_np = np.hstack((a,b,c)).T

data   = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).long()

partitions = [3*nPerClust-400,200,200]

train_data,devtest_data, train_labels,devtest_labels  = train_test_split(
    data,labels,train_size=partitions[0]
)

dev_data,test_data, dev_labels,test_labels = train_test_split(
    devtest_data,devtest_labels,test_size=partitions[1]
)

train_dataS = TensorDataset(train_data,train_labels)
dev_dataS   = TensorDataset(dev_data,dev_labels)
test_dataS  = TensorDataset(test_data,test_labels)

batchsize    = 32
train_loader = DataLoader(train_dataS, batch_size=batchsize,shuffle=True,drop_last=True)
dev_loader   = DataLoader(dev_dataS,batch_size=dev_dataS.tensors[0].shape[0])
test_loader  = DataLoader(test_dataS,batch_size=test_dataS.tensors[0].shape[0])

def createTheNet():

    class qwertyNet(nn.Module):

        def __init__(self):
            super().__init__()

            self.input = nn.Linear(2,4)

            self.fc1 = nn.Linear(4,64)

            self.output = nn.Linear(64,3)

        def forward(self,x):
            x = F.relu(self.input(x))
            x = F.relu(self.fc1(x))

            return self.output(x)

    net = qwertyNet()
    lossfn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.01)

    return net,lossfn,opt

def trainTheModel():

    theBestModel = {'Accuracy':0, 'net':None}

    numepochs = 200

    net,lossfn,opt = createTheNet()

    losses = torch.zeros(numepochs)
    trainAcc = []
    devAcc  = []

    for epochi in range(numepochs):

        net.train()

        batchAcc   = []
        batchLoss  = []
        for X,y in train_loader:

            yHat = net(X)
            loss = lossfn(yHat,y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            batchAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1) == y).float()))
            batchLoss.append( loss.item())

        trainAcc.append( np.mean(batchAcc) )
        losses[epochi] = np.mean(batchLoss)

        net.eval()

        X,y = next(iter(dev_loader))

        with torch.no_grad():
            yHat = net(X)

        devAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1) == y).float()) )

        if devAcc[-1]>theBestModel['Accuracy']:

            theBestModel['Accuracy'] = devAcc[-1].item()

            theBestModel['net'] = copy.deepcopy( net.state_dict() )

    return trainAcc,devAcc,losses,theBestModel

trainAcc,devAcc,losses,theBestModel = trainTheModel()
X,y = next(iter(test_loader))

bestnet = createTheNet()[0]
bestnet.load_state_dict(theBestModel['net'])

yHat = bestnet(X)

bestAcc = 100*torch.mean((torch.argmax(yHat,axis=1) == y).float())

fig = plt.figure(figsize=(10,5))

plt.plot(trainAcc,'o-',label='Train')
plt.plot(devAcc,'o-',label='Devset')
plt.plot([0,len(devAcc)],[bestAcc,bestAcc],'r--',label='Best dev model on TEST')
plt.ylabel('Accuracy (%)')
plt.xlabel('Epoch')
plt.title('Accuracy')
plt.ylim([bestAcc-5,bestAcc+5])
plt.legend()
plt.show()