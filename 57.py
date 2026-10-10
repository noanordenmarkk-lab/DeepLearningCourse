import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150

import seaborn as sns
iris = sns.load_dataset('iris')

# data = torch.tensor( iris[iris.columns[0:4]].values ).float()

# labels = torch.zeros(len(data), dtype=torch.long)
# labels[iris.species=='versicolor'] = 1
# labels[iris.species=='virginica'] = 2

# class ANNiris(nn.Module):
#     def __init__(self,nUnits,nLayers):
#         super().__init__()

#         self.layers = nn.ModuleDict()
#         self.nLayers = nLayers

#         self.layers['input'] = nn.Linear(4,nUnits)

#         for i in range(self.nLayers):
#             self.layers[f'hidden{i}'] = nn.Linear(nUnits,nUnits)

#         self.layers['output'] = nn.Linear(nUnits,3)

#     def forward(self,x):

#         x = self.layers['input'](x)

#         for i in range(self.nLayers):
#             x = F.relu( self.layers[f'hidden{i}'](x) )

#         x = self.layers['output'](x)
#         return x


# nUnitsPerLayer = 12
# nLayers = 4
# net = ANNiris(nUnitsPerLayer,nLayers)

# numepochs = 1000
# losses = torch.zeros(numepochs)

# def trainTheModel(theModel):

#     lossfun = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(theModel.parameters(),lr=.01)

#     for epochi in range(numepochs):

#         yHat = theModel(data)

#         loss = lossfun(yHat, labels)
#         losses[epochi] = loss

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     predictions = theModel(data)
#     predlabels = torch.argmax(predictions, axis=1)
#     acc = 100*torch.mean((predlabels == labels).float())

#     nParams = sum(p.numel() for p in theModel.parameters() if p.requires_grad)

#     return acc,nParams

# acc = trainTheModel(net)

# print(acc)


