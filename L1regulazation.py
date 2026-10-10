import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

import seaborn as sns
iris = sns.load_dataset('iris')

data = torch.tensor( iris[iris.columns[0:4]].values ).float()

labels = torch.zeros(len(data), dtype=torch.long)
labels[iris.species=='versicolor'] = 1
labels[iris.species=='virginica'] = 2

train_data,test_data, train_labels,test_labels = \
    train_test_split(data,labels, test_size=.2)

train_dataDataset = TensorDataset(train_data, train_labels)
test_dataDataset = TensorDataset(test_data,test_labels)

train_loader = DataLoader(train_dataDataset,batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataDataset,batch_size=test_dataDataset.tensors[0].shape[0])

def createANewModel():

    ANNiris = nn.Sequential(
        nn.Linear(4,64),
        nn.ReLU(),
        nn.Linear(64,64),
        nn.ReLU(),
        nn.Linear(64,3),
    )

    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ANNiris.parameters(), lr=.005)

    return ANNiris,lossfun,optimizer

numepochs = 1000

def trainTheModel(L2lambda):

    ANNiris,lossfun,optimizer = createANewModel()

    trainAcc = []
    testAcc = []
    losses = []

    nweights = 0
    for pname,weight in ANNiris.named_parameters():
        if 'bias' not in pname:
            nweights = nweights + weight.numel()

    for epochi in range(numepochs):

        batchAcc = []
        batchloss = []
        for X,y in train_loader:

            yHat = ANNiris(X)
            loss = lossfun(yHat, y)

            L2_term = torch.tensor(0.,requires_grad=True)

            for pname,weight in ANNiris.named_parameters():
                if 'bias' not in pname:
                    L2_term = L2_term + torch.sum(weight**2)

            loss = loss + L2lambda*L2_term/nweights

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1) == y).float()).item())
            batchloss.append( loss.item() )

            X,y = next(iter(test_loader))
            predlabels = torch.argmax( ANNiris(X),axis=1 )
            testAcc.append( 100*torch.mean((predlabels == y).float()).item() )

        trainAcc.append(np.mean(batchAcc))
        losses.append(np.mean(batchloss))

    return trainAcc,testAcc,losses


def smooth(x,k):
    return np.convolve(x,np.ones(k)/k,mode='same')

L2lambda = np.linspace(0,.005,10)

accResultsTrain = np.zeros((numepochs,len(L2lambda)))
accResultsTest = np.zeros((numepochs,len(L2lambda)))

for li in range(len(L2lambda)):

    ANNiris,lossfun,optimizer = createANewModel()
    trainAcc,testAcc,losses = trainTheModel(L2lambda[li])

    accResultsTrain[:,li] = smooth(trainAcc,10)
    accResultsTest[:,li] = smooth(testAcc,10)

fig,ax = plt.subplots(1,2,figsize=(17,7))

ax[0].plot(accResultsTrain)
ax[0].set_title('Train accuracy')
ax[1].plot(accResultsTest)
ax[1].set_title('Test Accuracy')

leglabels = [np.round(i,2) for i in L2lambda]

for i in range(2):
    ax[i].legend(leglabels)
    ax[i].set_xlabel('Epoch')
    ax[i].set_ylabel('Accuracy (%)')
    ax[i].set_ylim([50,101])
    ax[i].grid()

plt.show()