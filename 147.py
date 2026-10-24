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
    def __init__(self,):
        super().__init__()

        self.input  = nn.Linear(11,64)
        self.fc1    = nn.Linear(64,64)
        self.fc2    = nn.Linear(64,32)
        self.output = nn.Linear(32,1)

    def forward(self,x):
        x      = F.relu(self.input(x))
        x      = F.relu(self.fc1(x))
        x      = F.relu(self.fc2(x))
        return self.output(x)

numepochs = 300

def trainTheModel(winenet):

    #lossfunction and optimizer
    lossfun = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(winenet.parameters(),lr=.001)

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

def Compare_XvsK():
    fig,ax = plt.subplots(1,2,figsize=(17,7))
    #Kaiming-run 10 times

    K_losses = []
    K_Acc = []

    for i in range(11):
        winenetK = ANNwine()

        trainAcc,testAcc,losses = trainTheModel(winenetK)

        K_losses.append(torch.mean(losses[-10:]).item())
        K_Acc.append(np.mean(testAcc[-10:]).item())

    #plot or store the results
    ax[0].set_title('Loss')
    ax[0].plot(K_losses,color='b',label='Kaiming')

    ax[1].set_title('Accuracy')
    ax[1].plot(K_Acc,color='b')
    
    plt.legend()

    #Xavier-run 10 times

    X_losses = []
    X_Acc = []

    #Run it
    for i in range(11):
        winenetX = ANNwine()
         # change the weights (leave biases as Kaiming [default])
        for p in winenetX.named_parameters():
            if 'weight' in p[0]:
                nn.init.xavier_normal_(p[1].data)
        
        trainAcc,testAcc,losses = trainTheModel(winenetX)

        X_losses.append(torch.mean(losses[-10:]).item())
        X_Acc.append(np.mean(testAcc[-10:]).item())
        
    #plot or store the results
    ax[0].plot(X_losses,color='r',label='Xavier')
    ax[1].plot(X_Acc,color='r',label='Xavier')

    t_stat, p_value = stats.ttest_ind(
        K_losses,
        X_losses,
        equal_var=False
    )

    print("t-statistic:", t_stat)
    print("p-value:", p_value)

    plt.legend()
    plt.show()



Compare_XvsK()