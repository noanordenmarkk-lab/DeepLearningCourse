import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo 
import time

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

data = pd.read_csv(url,sep=';')

data = data[data['total sulfur dioxide']<200]

def dataLoaderMaker(qualtresh):
    dataf = data.copy()

    dataf['boolQuality'] = 0
    dataf.loc[dataf['quality'] > qualtresh, 'boolQuality'] = 1

    cols2zscore = dataf.keys()
    cols2zscore = cols2zscore.drop(['quality', 'boolQuality'])

    dataf[cols2zscore] = dataf[cols2zscore].apply(stats.zscore)

    dataT = torch.tensor(dataf[cols2zscore].values).float()
    labels = torch.tensor(dataf['boolQuality'].values).float()
    labels = labels[:, None]

    # use scikitlearn to split the data
    train_data,test_data, train_labels,test_labels = train_test_split(dataT, labels, test_size=.1)

    # then convert them into PyTorch Datasets (note: already converted to tensors)
    train_data = TensorDataset(train_data,train_labels)
    test_data  = TensorDataset(test_data,test_labels)

    batchsize    = 32
    train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True)
    test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

    return train_loader,test_loader

def createTheModel():

    class WineModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.input = nn.Linear(11,32)
            self.hidden1 = nn.Linear(32,32)
            self.output = nn.Linear(32,1)

        def forward(self,x):

           x = F.leaky_relu(self.input(x)) 
           x = F.leaky_relu(self.hidden1(x))

           return self.output(x)

    
    net = WineModel()

    learning_rate = 0.001
    lossfun = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate,weight_decay=.001)

    return net,lossfun,optimizer


def trainModel(qualtresh):
    train_loader,test_loader = dataLoaderMaker(qualtresh)
    net,lossfun,optimizer = createTheModel()

    numepochs = 500

    trainAcc = []
    testAcc = []
    losses = torch.zeros(numepochs)

    for epochi in range(numepochs):

        net.train()

        batchAcc = []
        batchLoss = []

        for X,y in train_loader:

          yHatTrain = net(X)
          loss = lossfun(yHatTrain,y)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          batchAcc.append(100*torch.mean(((yHatTrain > 0).float() == y).float()).item())
          batchLoss.append( loss.item() )

        trainAcc.append(np.mean(batchAcc))
        losses[epochi] = np.mean(batchLoss)

        net.eval()
        X,y = next(iter(test_loader))
        with torch.no_grad():
           yhatTest = net(X)
        testAcc.append(100 * torch.mean(((yhatTest > 0) == y).float()).item())

    return net,trainAcc,testAcc,losses,(yhatTest>0)

fig,ax = plt.subplots(3,3,figsize=(24,18))

#tresh 4,5
anet,atrainAcc,atestAcc,alosses,ayhatTest = trainModel(4.5)
ax[0,0].plot(alosses)
ax[0,0].set_title('losses with tresh 4.5')

ax[0,1].plot(atrainAcc,label='trainAcc')
ax[0,1].plot(atestAcc,label='testAcc')
ax[0,1].set_title('Accuracy with tresh 4.5')

acounts = torch.bincount(ayhatTest.int().flatten())

ax[0,2].bar([0, 1], acounts)
ax[0,2].set_xticks([0, 1])
ax[0,2].set_xticklabels(['Bad', 'Good'])
ax[0,2].set_ylabel('Num')

#tresh 5,5
bnet,btrainAcc,btestAcc,blosses,byhatTest = trainModel(5.5)
ax[1,0].plot(blosses)
ax[1,0].set_title('losses with tresh 5.5')

ax[1,1].plot(btrainAcc,label='trainAcc')
ax[1,1].plot(btestAcc,label='testAcc')
ax[1,1].set_title('Accuracy with tresh 5.5')

bcounts = torch.bincount(byhatTest.int().flatten())

ax[1,2].bar([0, 1], bcounts)
ax[1,2].set_xticks([0, 1])
ax[1,2].set_xticklabels(['Bad', 'Good'])
ax[1,2].set_ylabel('Num')

#tresh 6,5

cnet,ctrainAcc,ctestAcc,closses,cyhatTest = trainModel(6.5)
ax[2,0].plot(closses)
ax[2,0].set_title('losses with tresh 6.5')

ax[2,1].plot(ctrainAcc,label='trainAcc')
ax[2,1].plot(ctestAcc,label='testAcc')
ax[2,1].set_title('Accuracy with tresh 6.5')

ccounts = torch.bincount(cyhatTest.int().flatten())

ax[2,2].bar([0, 1], ccounts)
ax[2,2].set_xticks([0, 1])
ax[2,2].set_xticklabels(['Bad', 'Good'])
ax[2,2].set_ylabel('Num')

plt.legend()

plt.show()