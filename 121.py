import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

dataFull = np.loadtxt(
    open(r'C:\Users\Administrator\Documents\Programming\DeepLearningCourse\Datasets\mnist_train_small.csv', 'rb'),
    delimiter=','
)

def MakeDataset(N,doubleTheData=False):

    labels = dataFull[:N,0]
    data   = dataFull[:N,1:]

    dataNorm = data / np.max(data)

    if doubleTheData:
        dataNorm = np.concatenate((dataNorm,dataNorm),axis=0)
        labels   = np.concatenate((labels,labels),axis=0)

    dataT   = torch.tensor(dataNorm).float()
    labelsT = torch.tensor(labels).long()

    train_data,test_data, train_labels,test_labels = train_test_split(
        dataT,labelsT,train_size=.8
    )

    train_data = TensorDataset(train_data,train_labels)
    test_data  = TensorDataset(test_data,test_labels)

    batchsize = 20
    train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)
    test_loader  = DataLoader(test_data,batch_size=train_data.tensors[0].shape[0])

    return train_loader,test_loader

def createModel():

    class mnistnet(nn.Module):
        def __init__(self):
            super().__init__()

            self.input  = nn.Linear(784,64)
            self.fc1    = nn.Linear(64,32)
            self.fc2    = nn.Linear(32,32)
            self.output = nn.Linear(32,10)

        def forward(self,x):

            x = F.relu(self.input(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            return self.output(x)

    net = mnistnet()
    lossfun = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(net.parameters(),lr=.01)

    return net,lossfun,opt

def trainModel(N,doubleTheData=False):

    numepochs = 50
    train_loader,test_loader = MakeDataset(N,doubleTheData=False)
    net,lossfun,opt = createModel()

    trainAcc = []
    testAcc  = []
    losses   = torch.zeros(numepochs)

    for epochi in range(numepochs):

        batchAcc = []
        batchLoss = []
        for X,y in train_loader:

            yHat = net(X)
            loss = lossfun(yHat,y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            batchAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1) == y).float()) )
            batchLoss.append( loss.item() )

        trainAcc.append( np.mean(batchAcc))
        losses[epochi] = np.mean(batchLoss)

        net.eval()

        X,y = next(iter(test_loader))
        with torch.no_grad():
            yHat = net(X)

        testAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1) == y).float()) )

    return trainAcc,testAcc,losses,net


trainAcc,testAcc,losses,net = trainModel(5000)
train_loader,test_loader = MakeDataset(5000)

torch.save(net.state_dict(),'trainedModel.pt')

# create two new models of the same class
model1 = createModel()[0]
model2 = createModel()[0]

# replace one model's parameters with those of the trained net
model1.load_state_dict(torch.load('trainedModel.pt'))

# get some data from the test loader
X,y = next(iter(test_loader))

# run the data through both models
yHatNet    = net(X)
yHatModel1 = model1(X)
yHatModel2 = model2(X)


# show that the results overlap
fig = plt.figure(figsize=(12,5))
plt.plot(yHatNet[:,5].detach(),'b',label='Original')
plt.plot(yHatModel1[:,5].detach(),'ro',label='Loaded')
plt.plot(yHatModel2[:,5].detach(),'mx',label='Not loaded')
plt.legend()
plt.xlabel('Stimulus index')
plt.ylabel('Model output for node "6"')
plt.xlim([1000,1100])

plt.show()