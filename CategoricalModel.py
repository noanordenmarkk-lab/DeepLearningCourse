import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from IPython import display
plt.rcParams['figure.dpi'] = 150
def GetData():
    #model features
    learning_rate = .01

    #Create data
    nPerClust = 100
    blur = 1

    A = [ 1, 3 ]
    B = [ 1, -2 ]

    #Generate data
    a = [ A[0]+np.random.randn(nPerClust)*blur , A[1]+np.random.randn(nPerClust)*blur ]
    b = [ B[0]+np.random.randn(nPerClust)*blur , B[1]+np.random.randn(nPerClust)*blur ]

    #true lablels
    labels_np = np.vstack((np.zeros((nPerClust,1)),np.ones((nPerClust,1))))

    #concatanate into a matrix
    data_np = np.hstack((a,b)).T

    #convert to a pytorch tensor
    data = torch.tensor(data_np).float()
    labels = torch.tensor(labels_np).float()

    return data, labels, nPerClust, learning_rate


def CreateModel(learning_rate):
    #build the model
    ANNclassify = nn.Sequential(
        nn.Linear(2,16),
        # nn.ReLU(),
        nn.Linear(16,1),
        # nn.ReLU(),
        nn.Linear(1,1),
        nn.Sigmoid(),
        )

    #loss function
    lossfun = nn.BCELoss() #BCE = Binary cross entropy loss function
    
    #optimizer
    optimizer = torch.optim.SGD(ANNclassify.parameters(),
                                lr=learning_rate)

    #model output
    return ANNclassify, lossfun, optimizer

numepochs = 1000

def TrainModel():
    data, labels, nPerClust, learning_rate = GetData()

    ANNclassify, lossfun, optimizer = CreateModel(learning_rate)

    #train the model
    losses = torch.zeros(numepochs)

    for epochi in range(numepochs):

        #forward pass
        yHat = ANNclassify(data)

        #compute loss
        loss = lossfun(yHat, labels)
        losses[epochi] = loss

        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #compute the predictions
    predictions = ANNclassify(data)
    predlabels = predictions>.5

    #find errors
    misclassified = np.where(predlabels != labels)[0]

    #total accuracy
    totalacc = 100*torch.mean(((predictions>.5) == labels).float())

    return losses, predictions, totalacc, predlabels, misclassified

data, labels, nPerClust, learning_rate = GetData()

ANNclassify, lossfun, optimizer = CreateModel(learning_rate)

losses, predictions, totalacc, predlabels, misclassified = TrainModel()

#learning rates
learningrates = np.linspace(.001, .1,50)

#initialize
accByLR = []
allLosses = np.zeros((len(learningrates), numepochs))

#the loop
for i,lr in enumerate(learningrates):

    #Create and run the model
    ANNclassify, lossfun, optimizer = CreateModel(lr)
    losses, predictions, totalacc, predlabels, misclassified = TrainModel()

    #store the results
    accByLR.append(totalacc)
    allLosses[i,:] = losses.detach()

fig,ax = plt.subplots(1,2,figsize=(16,4))

ax[0].plot(learningrates,accByLR,'s-')
ax[0].set_xlabel('Learning rate')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Accuracy by learning rate')

ax[1].plot(allLosses.T)
ax[1].set_title('Losses by learning rate')
ax[1].set_xlabel('Epoch number')
ax[1].set_ylabel('Loss')
plt.show()


# fig = plt.figure(figsize=(5,5))
# plt.plot(data[misclassified,0],data[misclassified,1],'rx',markersize=12,markeredgewidth=10)
# plt.plot(data[np.where(~predlabels)[0],0],data[np.where(~predlabels)[0],1],'bs')
# plt.plot(data[np.where(predlabels)[0],0],data[np.where(predlabels)[0],1],'ko')

# plt.legend(['Misclassified', 'blue', 'black'],bbox_to_anchor=(1,1))
# plt.title(f'{totalacc}% correct')
# plt.show()

