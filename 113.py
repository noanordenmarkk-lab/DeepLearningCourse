import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = np.loadtxt(
    open(r'C:\Users\Administrator\Documents\Programming\DeepLearningCourse\Datasets\mnist_train_small.csv', 'rb'),
    delimiter=','
)

labels = data[:,0]
data   = data[:,1:]

#normalize the data to a range of 0 to 1
dataNorm = data / np.max(data)

#scramble the images
eggs      = np.random.permutation(data.shape[1])
scrambled = dataNorm[:,eggs]

#convert to tensor
dataT   = torch.tensor(scrambled).float()
labelsT = torch.tensor(labels).long() #long = int64

#use scikitlearn to split the data
train_data,test_data, train_labels,test_labels = train_test_split(dataT,labelsT, test_size=.1)

#convert into Pytorch datasets
train_data = TensorDataset(train_data,train_labels)
test_data  = TensorDataset(test_data,test_labels)

#translate into dataloader objects
batchsize    = 32
train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)
test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

def createModel():

    class eggnet(nn.Module):
        def __init__(self):
            super().__init__()

            self.input  = nn.Linear(784,64)
            self.fc1    = nn.Linear(64,64)
            self.fc2    = nn.Linear(64,32)
            self.output = nn.Linear(32,10)

        def forward(self,x):

            x = F.relu(self.input(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            return self.output(x)

    net       = eggnet()
    lossfun   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=.01)

    return net,lossfun,optimizer

def trainModel():

    net,lossfun,optimizer = createModel()

    numepochs = 100

    trainAcc = []
    testAcc  = []
    losses   = torch.zeros(numepochs)

    for epochi in range(numepochs):

        batchAcc  = []
        batchLoss = []
        for X,y in train_loader:

            yHat = net(X)
            loss = lossfun(yHat,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1) == y).float()) )
            batchLoss.append( loss.item() )

        trainAcc.append(np.mean(batchAcc))
        losses[epochi] = np.mean(batchLoss)

        net.eval()

        X,y = next(iter(test_loader))
        with torch.no_grad():
            yHat = net(X)

        testAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1) == y).float()) )

    return trainAcc,testAcc,losses,net

trainAcc,testAcc,losses,net = trainModel()

fig,ax = plt.subplots(1,1,figsize=(17,7))

ax.plot(trainAcc,label='train',color='b')
ax.plot(testAcc,label='test',color='r')

plt.legend()
plt.show()
