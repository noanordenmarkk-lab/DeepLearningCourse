import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from IPython import display
plt.rcParams['figure.dpi'] = 150

import seaborn as sns
iris = sns.load_dataset('iris')

# sns.pairplot(iris, hue='species')
# plt.show()

#convert from pandas dataframe to tensor
data = torch.tensor( iris[iris.columns[0:4]].values ).float()

labels = torch.zeros(len(data), dtype=torch.long)
#setosa in going to be mapped to the value of 0
labels[iris.species=='versicolor'] = 1
labels[iris.species=='virginica'] = 2

def CreateIrisModel(nHidden):
    #model architecture
    ANNiris = nn.Sequential(
        nn.Linear(4, nHidden),
        nn.ReLU(),
        nn.Linear(nHidden,nHidden),
        nn.ReLU(),
        nn.Linear(nHidden,3),
        )

    #loss function
    lossfun = nn.CrossEntropyLoss()

    #optimizer
    optimizer = torch.optim.SGD(ANNiris.parameters(),lr=0.1)

    return ANNiris, lossfun, optimizer

numepochs = 1000

def TrainModel(ANNiris):
    ANNiris, lossfun, optimizer = CreateIrisModel(nHidden)

    #initialize losses
    losses = torch.zeros(numepochs)
    ongoingAcc = []

    #loop over epochs
    for epochi in range(numepochs):

        yHat = ANNiris(data)

        loss = lossfun(yHat, labels)
        losses[epochi] = loss

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

#compute accuracy
matches = torch.argmax(yHat,axis=1) == labels
matchesNumeric = matches.float()
accuracyPct = 100*torch.mean(matchesNumeric)
ongoingAcc.append( accuracyPct )

#final forward pass
predictions = ANNiris(data)
predlabels = torch.argmax(predictions, axis=1)
totalacc = 100*torch.mean((predlabels == labels).float())

print('Final accuracy: %g%%' % totalacc)
print(nModelLayers)

fog, ax = plt.subplots(1,2,figsize=(13,4))

ax[0].plot(losses.detach())
ax[0].set_xlabel('Loss')
ax[0].set_ylabel('epoch')
ax[0].set_title('Losses')

ax[1].plot(ongoingAcc)
ax[1].set_ylabel('accuracy')
ax[1].set_xlabel('epoch')
ax[1].set_title('Accuracy')
plt.show()