import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from IPython import display
plt.rcParams['figure.dpi'] = 150

#Create data

nPerClust = 100
blur = 1

A = [ 1, 5 ]
B = [ 5, 5 ]
C = [ 2, 1 ]

#Generate Data
a = [A[0]+np.random.randn(nPerClust)*blur , A[1]+np.random.randn(nPerClust)*blur]
b = [B[0]+np.random.randn(nPerClust)*blur , B[1]+np.random.randn(nPerClust)*blur]
c = [C[0]+np.random.randn(nPerClust)*blur , C[1]+np.random.randn(nPerClust)*blur]

#concantrete into a matric
data_np = np.hstack((a,b,c)).T

#make it into a pandas dataframe
df = pd.DataFrame(data_np)

#convert from pandas dataframe to tesnsor
data = torch.tensor(df.iloc[:, 0:3].values).float()

#give each cetegory a number
labels = torch.zeros(len(data), dtype=torch.long)
labels[0:100] = 0
labels[100:200] = 1
labels[200:300] = 2

#model Archetecture
ANNQwerties = nn.Sequential(
    nn.Linear(2, 4), 
    nn.ReLU(),
    nn.Linear(4,4),
    nn.ReLU(),
    nn.Linear(4,3),
    nn.Softmax(dim=1)
)

#Loss function
lossfun = nn.CrossEntropyLoss()

#optimizer 
optimizer = torch.optim.SGD(ANNQwerties.parameters(),lr=0.1)

numepochs = 1000

losses = torch.zeros(numepochs)
ongoingAcc = []

#loop over epochs
for epochi in range(numepochs):

    yHat = ANNQwerties(data)

    loss = lossfun(yHat, labels)
    losses[epochi] = loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    matches = torch.argmax(yHat,axis=1) == labels
    matchesNumeric = matches.float()
    accuracyPct = 100*torch.mean(matchesNumeric)
    ongoingAcc.append( accuracyPct)

predictions = ANNQwerties(data)
predlabels = torch.argmax(predictions, axis = 1)
totalacc = 100*torch.mean((predlabels == labels).float())

print('Final accuracy: %g%%' % totalacc)

fig,ax = plt.subplots(1,1,figsize=(6,6))
ax.scatter(a[0], a[1], color='blue')
ax.scatter(b[0], b[1], color='green')
ax.scatter(c[0], c[1], color='red')

wrong = predlabels != labels

# Plot an X on them
ax.scatter(
    data[wrong, 0],
    data[wrong, 1],
    marker='x',
    color='black',
    s=100
)

plt.show()


