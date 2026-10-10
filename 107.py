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

#convert to tensor
dataT   = torch.tensor(data).float()
labelsT = torch.tensor(labels).long() #long = int64

#use scikitlearn to split the data
train_data,test_data, train_labels,test_labels = train_test_split(dataT,labelsT, test_size=.1)

#convert into Pytorch datasets
train_data = train_data/torch.max(train_data)
print('Training data range %g to %g'%(torch.min(train_data),torch.max(train_data)))
train_data = TensorDataset(train_data,train_labels)

test_data = test_data/torch.max(test_data)
print('Test data range %g to %g'%(torch.min(test_data),torch.max(test_data)))
test_data  = TensorDataset(test_data,test_labels)



#translate into dataloader objects
batchsize    = 32
train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)
test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

def createTheMNISTNet(nUnits,nLayers):

    class mnistNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.layers  = nn.ModuleDict()
            self.nLayers = nLayers

            self.layers['input'] = nn.Linear(784,nUnits)

            for i in range(nLayers):
                self.layers[f'hidden{i}'] = nn.Linear(nUnits,nUnits)

            self.layers['output'] = nn.Linear(nUnits,10)

        def forward(self,x):
            x = F.relu( self.layers['input'](x))

            for i in range(self.nLayers):
                x = F.relu( self.layers[f'hidden{i}'](x))

            return torch.log_softmax( self.layers['output'](x),axis=1 )
    
    net       = mnistNet()
    lossfun   = nn.NLLLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=.01)

    return net,lossfun,optimizer




def trainModel(nUnits,nLayers):

    numepochs = 60

    net,lossfun,optimizer = createTheMNISTNet(nUnits,nLayers)

    losses   = torch.zeros(numepochs)
    trainAcc = []
    testAcc  = []

    for epochi in range(numepochs):

        batchAcc  = []
        batchLoss = []
        for X,y in train_loader:

            yHat = net(X)
            loss = lossfun(yHat,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchLoss.append( loss.item() )
            batchAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1) == y).float()) )

        trainAcc.append( np.mean(batchAcc))
        losses[epochi] = np.mean(batchLoss)

        X,y = next(iter(test_loader))
        with torch.no_grad():
            yHat = net(X)

        testAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1)==y).float()) )

    return trainAcc,testAcc,losses,net

fig,ax = plt.subplots(1,2,figsize=(16,5))

nLayers = [1,2,3]
nUnits  = torch.arange(50,275,25)

for i in range(len(nLayers)):

    trainAccs = []
    testAccs = []

    for i2 in range(len(nUnits)):

        trainAcc,testAcc,losses,net = trainModel(nUnits[i2],nLayers[i])

        trainAccs.append(np.mean(trainAcc[-10:]))
        testAccs.append(np.mean(testAcc[-10:]))

    ax[0].plot(nUnits, trainAccs, label=str(nLayers[i]))
    ax[1].plot(nUnits, testAccs, label=str(nLayers[i]))

ax[0].set_xlabel('Number of hidden units')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Train')

ax[1].set_xlabel('NUmber of hidden layers')
ax[1].set_ylabel('Accuracy')
ax[1].set_title('Test')
ax[1].legend()

plt.show()