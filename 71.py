import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
plt.rcParams['figure.dpi'] = 150

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset

#Create data

nPerClust = 200

th = np.linspace(0,4*np.pi,nPerClust)
r1 = 10
r2 = 15

#generate data
a = [ r1*np.cos(th) + np.random.randn(nPerClust)*3 , 
    r1*np.sin(th) + np.random.randn(nPerClust)  ]

b = [ r2*np.cos(th) + np.random.randn(nPerClust) , 
    r2*np.sin(th) + np.random.randn(nPerClust)*3 ]

#true labels
labels_np = np.vstack((np.zeros((nPerClust,1)),np.ones((nPerClust,1))))

#concentrate into matrix
data_np = np.hstack((a,b)).T

#convert to a pytorch tensor
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()

#Split the data
train_data,test_data, train_labels,test_labels = train_test_split(data,labels,test_size=.2)

#Convert to pytorch dataset
train_data = TensorDataset(train_data,train_labels)
test_data = TensorDataset(test_data,test_labels)

#translate into dataloader objects
batchsize = 16
train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True)
test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

#Create the model
class theModelClass(nn.Module):
    def __init__(self,dropoutRate):
        super().__init__()

        #layers
        self.input = nn.Linear(  2,128)
        self.hidden = nn.Linear(128,128)
        self.output = nn.Linear(128,  1)

        #parameters
        self.dr = dropoutRate

    #forward pass
    def forward(self,x):

        x = F.relu(self.input(x))

        x = F.dropout(x,p=self.dr,training=self.training)

        x = F.relu(self.hidden(x))

        x = F.dropout(x,p=self.dr,training=self.training)

        x = self.output(x)

        return x

def createANewModel(dropoutrate):

    #grab a instance of the model class
    ANNQC = theModelClass(dropoutrate)

    #loss function
    lossfun = nn.BCEWithLogitsLoss()

    #optimixer
    optimizer = torch.optim.SGD(ANNQC.parameters(),lr=.002)

    return ANNQC,lossfun,optimizer

#train the model

numepochs = 1000

def trainTheModel(ANNQC,lossfun,optimizer):

    trainAcc = []
    testAcc = []

    for epochi in range(numepochs):

        ANNQC.train()

        batchAcc = []
        for X,y in train_loader: #Data X, labels y

            yHat = ANNQC(X)
            loss = lossfun(yHat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #computing training acc just for this batch
            batchAcc.append( 100*torch.mean(((yHat>.5)==y).float()).item() )

        trainAcc.append( np.mean(batchAcc) )

        ANNQC.eval() #switch training off
        X,y = next(iter(test_loader))
        yHat = ANNQC(X)
        testAcc.append( 100*torch.mean(((yHat>.5)==y).float()).item() )

    return trainAcc,testAcc

dropoutrate = .0
ANNQC,lossfun,optimizer = createANewModel(dropoutrate)

trainAcc,testAcc = trainTheModel(ANNQC,lossfun,optimizer)

def smooth(x,k=5):
    return np.convolve(x,np.ones(k)/k,mode='same')

fig = plt.figure(figsize=(10,5))

plt.plot(smooth(trainAcc),'bs-')
plt.plot(smooth(testAcc),'ro-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend(['Train','Test'])
plt.title('Dropout rate = %g'%dropoutrate)

plt.show()