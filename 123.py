import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

nPerClust = 300
blur = 1

A = [ 1, 1 ]
B = [ 5, 1 ]
C = [ 4, 3 ]

a = [ A[0]+np.random.randn(nPerClust)*blur , A[1]+np.random.randn(nPerClust)*blur]
b = [ B[0]+np.random.randn(nPerClust)*blur , B[1]+np.random.randn(nPerClust)*blur]
c = [ C[0]+np.random.randn(nPerClust)*blur , C[1]+np.random.randn(nPerClust)*blur]

labels_np = np.hstack(( np.zeros((nPerClust)),
                        np.ones( (nPerClust)),
                      1+np.ones( (nPerClust)) ))

data_np = np.hstack((a,b,c)).T

data   = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).long()

dist2orig = torch.sqrt( data[:,0]**2+data[:,1]**2 )

dataAug   = torch.cat((data,dist2orig.view(len(data),1)),axis=1)

train_data,test_data, train_labels,test_labels = train_test_split(
    dataAug, labels, test_size=.2
)

train_data = TensorDataset(train_data,train_labels)
test_data  = TensorDataset(test_data,test_labels)

batchsize    = 16
train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)
test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

def createTheNet(useExtraFeature=False):

    class qwertyNet(nn.Module):

        def __init__(self):
            super().__init__()

            if useExtraFeature:
                self.input = nn.Linear(3,8)
            else:
                self.input = nn.Linear(2,8)

            self.fc1 = nn.Linear(8,8)

            self.output = nn.Linear(8,3)

        def forward(self,x):

            if not useExtraFeature:
                x = x[:,:2]
            x = F.relu(self.input(x))
            x = F.relu(self.fc1(x))

            return self.output(x)

    net = qwertyNet()
    lossfn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(net.parameters(),lr=.001)

    return net,lossfn,opt

def trainTheModel(useExtraFeature=False):

    numepochs = 1000

    net,lossfn,opt = createTheNet(useExtraFeature)

    losses = torch.zeros(numepochs)
    trainAcc = []
    testAcc  = []

    for epochi in range(numepochs):

        batchAcc = []
        batchLoss  = []
        for X,y in train_loader:

            yHat = net(X)
            loss = lossfn(yHat,y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            batchAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1) == y).float()))
            batchLoss.append( loss.item())

        trainAcc.append( np.mean(batchAcc) )
        losses[epochi] = np.mean(batchLoss)

        net.eval()

        X,y = next(iter(test_loader))

        with torch.no_grad():
            yHat = net(X)

        testAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1) == y).float()) )

        net.train()

    return trainAcc,testAcc,losses,net

def plotTheResults(trainAcc,testAcc,losses,net):

  # compute accuracy over entire dataset (train+test)
  yHat = net(dataAug)
  predictions = torch.argmax(yHat,axis=1)
  accuracy = (predictions == labels).float()

  # and accuracy by group
  accuracyByGroup = np.zeros(3)
  for i in range(3):
    accuracyByGroup[i] = 100*torch.mean(accuracy[labels==i])


  # create the figure
  fig,ax = plt.subplots(2,2,figsize=(10,6))

  # plot the loss function
  ax[0,0].plot(losses.detach())
  ax[0,0].set_ylabel('Loss')
  ax[0,0].set_xlabel('epoch')
  ax[0,0].set_title('Losses')

  # plot the accuracy functions
  ax[0,1].plot(trainAcc,label='Train')
  ax[0,1].plot(testAcc,label='Test')
  ax[0,1].set_ylabel('Accuracy (%)')
  ax[0,1].set_xlabel('Epoch')
  ax[0,1].set_title('Accuracy')
  ax[0,1].legend()

  # plot overall accuracy by group
  ax[1,0].bar(range(3),accuracyByGroup)
  ax[1,0].set_ylim([np.min(accuracyByGroup)-5,np.max(accuracyByGroup)+5])
  ax[1,0].set_xticks([0,1,2])
  ax[1,0].set_xlabel('Group')
  ax[1,0].set_ylabel('Accuracy (%)')
  ax[1,0].set_title('Accuracy by group')

  # scatterplot of correct and incorrect labeled data
  colorShapes = [ 'bs','ko','g^' ] # data markers
  for i in range(3):
    # plot all data points
    ax[1,1].plot(dataAug[labels==i,0],dataAug[labels==i,1],colorShapes[i],
                 alpha=.3,label=f'Group {i}')
    
    # cross-out the incorrect ones
    idxErr = (accuracy==0) & (labels==i)
    ax[1,1].plot(dataAug[idxErr,0],dataAug[idxErr,1],'rx')

  ax[1,1].set_title('All groups')
  ax[1,1].set_xlabel('qwerty dimension 1')
  ax[1,1].set_ylabel('qwerty dimension 2')
  ax[1,1].legend()
  
  plt.tight_layout()
  plt.show()

finalacc2 = np.zeros(10)
finalacc3 = np.zeros(10)

for i in range(10):
  finalacc2[i] = trainTheModel(False)[1][-1]
  finalacc3[i] = trainTheModel(True)[1][-1]

# show the numbers
print( np.round(np.vstack((finalacc2,finalacc3)).T,2) )

# run the t-test and print the results
from scipy import stats
t,p = stats.ttest_ind(finalacc3,finalacc2)
print('\n\nt=%.2f, p=%.2f' %(t,p))