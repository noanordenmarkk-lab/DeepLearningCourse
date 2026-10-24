import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scipy.stats as stats

data = np.loadtxt(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    delimiter=",",
    dtype=str
)

mask = ~(data == '?').any(axis=1)

data = data[mask]

labels = data[:,-1].astype(float)
data   = data[:,:-1].astype(float)

dataf = pd.DataFrame(data)

#binarize the data
dataf['boolDiagnosis'] = 0
dataf.loc[labels > 0, 'boolDiagnosis'] = 1

cols2zscore = dataf.keys()
cols2zscore = cols2zscore.drop(['boolDiagnosis'])

dataf[cols2zscore] = dataf[cols2zscore].apply(stats.zscore)

#convert to tensor
dataT   = torch.tensor(dataf[cols2zscore].values,dtype=torch.float32)
labelsT = torch.tensor(dataf['boolDiagnosis'].values,dtype=torch.float32)
labelsT = labelsT[:, None]

#use scikitlearn to split the data
train_data,test_data, train_labels,test_labels = train_test_split(dataT,labelsT, test_size=.3)

#convert into Pytorch datasets
train_data = TensorDataset(train_data,train_labels)
test_data  = TensorDataset(test_data,test_labels)

#translate into dataloader objects
batchsize    = 16
train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)
test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

def createTheModel():

    class HeartDiseaseNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.input  = nn.Linear(13,8)
            self.fc1    = nn.Linear(8,8)
            self.output = nn.Linear(8,1)

        def forward(self,x):

            x = F.relu(self.input(x))
            x = F.relu(self.fc1(x))

            return torch.sigmoid(self.output(x))

    net = HeartDiseaseNet()
    lossfun = nn.BCELoss()
    optim = torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.001)

    return net,lossfun,optim

def trainTheModel():

    numepochs = 50

    net,lossfun,optim = createTheModel()

    trainAcc = []
    testAcc  = []
    losses = torch.zeros(numepochs)

    for epochi in range(numepochs):

        net.train()
        
        batchLoss = []
        batchAcc  = []
        for X,y in train_loader:

            yHat = net(X)
            loss = lossfun(yHat,y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            batchAcc.append(100*torch.mean(((yHat > 0.5) == y).float()).item())
            batchLoss.append( loss.item() )

        trainAcc.append( np.mean(batchAcc) )
        losses[epochi] = np.mean( batchLoss )

        net.eval()

        X,y = next(iter(test_loader))
        with torch.no_grad():
            yHat = net(X)

        testAcc.append(100*torch.mean(((yHat > 0.5) == y).float()).item())
    return trainAcc,testAcc,losses,net

trainAcc,testAcc,losses,net = trainTheModel()

fig,ax = plt.subplots(1,2,figsize=(17,7))

ax[0].plot(losses,label='Loss')
ax[0].set_title('Loss')

ax[1].plot(trainAcc,label='Train Accuracy')
ax[1].plot(testAcc,label='Test Accuracy')
ax[1].set_title('Accuracy')

plt.legend()
plt.show()