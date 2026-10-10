import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np
import scipy.stats as stats

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib_inline
import matplotlib
matplotlib.use("QtAgg")

url  = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url,sep=';')
data = data[data['total sulfur dioxide']<200]

#z-score normalization
col2zscore       = data.keys()
col2zscore       = col2zscore.drop('quality')
data[col2zscore] = data[col2zscore].apply(stats.zscore)

data['boolQuality'] = 0
data.loc[data['quality'] > 5, 'boolQuality'] = 1

#convert to Tensor
dataT  = torch.tensor( data[col2zscore].values ).float()
labels = torch.tensor( data['boolQuality'].values ).float()
labels = labels[:,None]

#Use scikitlearn to split the data
train_data,test_data, train_labels,test_labels = \
train_test_split(dataT,labels,test_size=.2)

#convert the, into Pythorch Datasets
train_dataDataset = TensorDataset(train_data,train_labels)
test_dataDataset  = TensorDataset(test_data,test_labels)

#train and test loaders
train_loader = DataLoader(train_dataDataset, shuffle=True, batch_size=32, drop_last=True)
test_loader  = DataLoader(test_dataDataset, batch_size=test_dataDataset.tensors[0].shape[0])

# create a class for the model

class ANNwine(nn.Module):
    def __init__(self,actfun):
        super().__init__()

        self.input  = nn.Linear(11,64)
        self.fc1    = nn.Linear(64,64)
        self.fc2    = nn.Linear(64,32)
        self.output = nn.Linear(32,1)
        self.actfun = actfun

    def forward(self,x):
        #get activation funtion type
        #this code replaces torch.relu with torch.<self.actfun>
        actfun = getattr(torch.nn, self.actfun)()
        x      = actfun(self.input(x))
        x      = actfun(self.fc1(x))
        x      = actfun(self.fc2(x))
        return self.output(x)

numepochs = 1000

def trainTheModel(actfun):

    winenet = ANNwine(actfun)

    #lossfunction and optimizer
    lossfun = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(winenet.parameters(),lr=.01)

    #initialize losses
    losses   = torch.zeros(numepochs)
    trainAcc = []
    testAcc  = []

    #loop over epochs
    for epochi in range(numepochs):

        batchAcc  = []
        batchLoss = []
        for X,y in train_loader:

            yHat = winenet(X)
            loss = lossfun(yHat,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchAcc.append(100*((yHat > 0) == y).float().mean().item())
            batchLoss.append( loss.item() )
            #end of batch loop

        trainAcc.append( np.mean(batchAcc) )

        losses[epochi] = np.mean(batchLoss)

        #test acc
        winenet.eval()
        X,y = next(iter(test_loader))
        with torch.no_grad():
            yHat = winenet(X)
        testAcc.append( 100*torch.mean(((yHat > 0) == y).float()).item())

    return trainAcc,testAcc,losses


activation_funs = ['ReLU', 'ReLU6', 'LeakyReLU']

trainByAct = np.zeros((numepochs,len(activation_funs)))
testByAct  = np.zeros((numepochs,len(activation_funs)))

for ai,actfun in enumerate(activation_funs):
    winenet = ANNwine(actfun)
    trainByAct[:,ai],testByAct[:,ai],losses = trainTheModel(actfun)

#plot some results
fig,ax = plt.subplots(1,2,figsize=(20,7))

ax[0].plot(trainByAct)
ax[0].set_title('train acc')
ax[1].plot(testByAct)
ax[1].set_title('test acc')

#common features
for i in range(2):
    ax[i].legend(activation_funs)
    ax[i].set_xlabel('Epoch')
    ax[i].set_ylabel('Acc (%)')
    ax[i].grid()

plt.show()