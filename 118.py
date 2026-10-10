import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

def createSomeData(nPerClust):

    A = [ 1, 1 ]
    B = [ 5, 1 ]
    C = [ 4, 4 ]

    a = [ A[0]+np.random.randn(nPerClust) , A[1]+np.random.randn(nPerClust)]
    b = [ B[0]+np.random.randn(nPerClust) , B[1]+np.random.randn(nPerClust)]
    c = [ C[0]+np.random.randn(nPerClust) , C[1]+np.random.randn(nPerClust)]

    labels_np = np.hstack(( np.zeros((nPerClust)),
                        np.ones( (nPerClust)),
                      1+np.ones( (nPerClust)) ))

    data_np = np.hstack((a,b,c)).T

    output  = {}
    output['data']   = torch.tensor(data_np).float()
    output['labels'] = torch.tensor(labels_np).long()

    #use sickitlearn to split the data
    train_data,test_data, train_labels,test_labels = train_test_split(output['data'],output['labels'],test_size=0.2)

    #then convert them into pytorch datasets
    train_data = TensorDataset(train_data,train_labels)
    test_data  = TensorDataset(test_data,test_labels)

    #finally, translate into dataloaders
    batch_size = 8
    output['data']    = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    output['labels']  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0],shuffle=False)

    return output

def createTheQwertyNet(nUnits,nLayers):

    class qwertnet(nn.Module):
        def __init__(self,nUnits,nLayers):
            super().__init__()

            self.Layers  = nn.ModuleDict()
            self.nLayers = nLayers

            self.Layers['input'] = nn.Linear(2,nUnits)

            for i in range(nLayers):
                self.Layers[f'hidden{i}'] = nn.Linear(nUnits,nUnits)

            self.Layers['output'] = nn.Linear(nUnits,3)

        def forward(self,x):

            x = F.relu(self.Layers['input'](x))

            for i in range(nLayers):
                    x = F.relu(self.Layers[f'hidden{i}'](x))

            return self.Layers['output'](x)

    net = qwertnet(nUnits,nLayers)
    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=0.01)

    return net,lossfun,optimizer

def trainTheModel(nUnits,nLayers,nPerclust):

    numepochs = 50

    net,lossfun,optimizer = createTheQwertyNet(nUnits,nLayers)
    output = createSomeData(nPerclust)

    trainAcc = []
    testAcc  = []

    losses = torch.zeros(numepochs)

    for epochi in range(numepochs):

        batchAcc  = []
        batchLoss = []
        for X,y in output['data']:

            yHat = net(X)
            loss = lossfun(yHat,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1) == y).float()) )
            batchLoss.append( loss.item() )

        trainAcc.append( np.mean(batchAcc))
        losses[epochi] = np.mean(batchLoss)

        net.eval()

        X,y = next(iter(output['labels']))
        with torch.no_grad():
            yHat = net(X)

        testAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1) == y).float()) )

    return net,trainAcc,testAcc,losses

net,trainAcc,testAcc,losses = trainTheModel(80,1,200)

nNodesInModel = 80
layersRange   = [ 1,5,10,20 ]
nDatapoints   = np.arange(50,551,50)

results = np.zeros((len(nDatapoints),len(layersRange),2))

for didx,pnts in enumerate(nDatapoints):

    for lidx,layers in enumerate(layersRange):

        unitsperlayer = int(nNodesInModel/layersRange[lidx])
        net,trainAcc,testAcc,losses = trainTheModel(unitsperlayer,layers,pnts)

        results[didx,lidx,0] = np.mean( testAcc[-5:] )
        results[didx,lidx,1] = torch.mean(losses[-5:]).item()

