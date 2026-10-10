import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from IPython import display
plt.rcParams['figure.dpi'] = 150

nPerClust = 100
blur = 1

A = [ 1, 1]
B = [ 5, 1]

a = [A[0]+np.random.randn(nPerClust)*blur, A[1]+np.random.randn(nPerClust)*blur]
b = [B[0]+np.random.randn(nPerClust)*blur, B[1]+np.random.randn(nPerClust)*blur]

labels_np = np = np.vstack((np.zeros((nPerClust,1)), np.ones((nPerClust))))

data_np = np.hstack((a,b)).T

data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()

class theClass4ANN(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(2,1)

        self.output = nn.Linear(1,1)

    def forward(self,x):

        x = self.input(x)

        x = F.relu(x)

        x = self.output(x)
        x = torch.sigmoid(x)

        return x

ANNclassify = theClass4ANN()

LR = .01

lossfun = nn.MSELoss()

optimizer = torch.optim.SGD(ANNclassify.parameters(), lr=LR)

numepochs = 1000
losses = torch.zeros(numepochs)

for epochi in range(numepochs):

    yHat = ANNclassify(data)

    loss = lossfun(yHat, labels)
    losses[epochi] = loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

predictions = ANNclassify(data)

predlabels = predictions > 0.5

misclassified = np.where(predlabels != labels)[0]

totalacc = 100-100*len(misclassified)/(2*nPerClust)

