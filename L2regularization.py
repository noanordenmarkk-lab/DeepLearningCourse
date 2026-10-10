import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader,TensorDataset

import seaborn as sns
iris = sns.load_dataset('iris')

data = torch.tensor( iris[iris.columns[0:4]].values ).float()

labels = torch.zeros(len(data), dtype=torch.long)

labels[iris.species=='versicolor'] = 1
labels[iris.species=='virginica'] = 2

train_data,test_data, train_labels,test_labels = \
    train_test_split(data,labels, test_size=.2)

train_dataDataset = TensorDataset(train_data,train_labels)
test_dataDataset = TensorDataset(test_data,test_labels)

train_loader = DataLoader(train_dataDataset,batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataDataset, batch_size=test_dataDataset.tensors[0].shape[0])

def createANewModel(L2lambda):

    #model architecture
    ANNiris = nn.Sequential(
        nn.Linear(4,64),
        nn.ReLU(),
        nn.Linear(64,64),
        nn.ReLU(),
        nn.Linear(64,3),
        )

    lossfun = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(ANNiris.parameters(),lr=0.005,weight_decay=L2lambda)

    return ANNiris,lossfun,optimizer

numepochs = 1000

def trainTheModel(ANNiris,lossfun,optimizer):

    trainAcc = []
    testAcc = []
    losses = []

    for epochi in range(numepochs):

        batchAcc = []
        batchLoss = []

        for X,y in train_loader:

            yHat = ANNiris(X)
            loss = lossfun(yHat,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1) == y).float()).item() )
            batchLoss.append( loss.item() )

        trainAcc.append(np.mean(batchAcc))
        losses.append(np.mean(batchLoss))

        X,y = next(iter(test_loader))
        predlabels = torch.argmax( ANNiris(X),axis=1 )
        testAcc.append( 100*torch.mean((predlabels == y).float()).item() )

    return trainAcc,testAcc, losses


L2lambda = 0.01
ANNiris,lossfun,optimizer = createANewModel(L2lambda)

trainAcc,testAcc, losses = trainTheModel(ANNiris,lossfun,optimizer)

def smooth(x,k=5):
    return np.convolve(x,np.ones(k)/k,mode='same')

l2lambdas = np.linspace(0,.1,10)

accResultsTrain = np.zeros((numepochs,len(l2lambdas)))
accResultsTest = np.zeros((numepochs,len(l2lambdas)))

for li in range(len(l2lambdas)):

    ANNiris,lossfun,optimizer = createANewModel(l2lambdas[li])
    trainAcc,testAcc,losses = trainTheModel(ANNiris,lossfun,optimizer)

    accResultsTrain[:,li] = smooth(trainAcc,10)
    accResultsTest[:,li] = smooth(testAcc,10)

fig,ax = plt.subplots(1,2,figsize=(17,7))

ax[0].plot(accResultsTrain)
ax[0].set_title('Train accuracy')
ax[1].plot(accResultsTest)
ax[1].set_title('Test Accuracy')

leglabels = [np.round(i,2) for i in l2lambdas]

for i in range(2):
    ax[i].legend(leglabels)
    ax[i].set_xlabel('Epoch')
    ax[i].set_ylabel('Accuracy (%)')
    ax[i].set_ylim([50,101])
    ax[i].grid()

plt.show()