import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#Create data

nDots = 100
blur = 1

X = [ 1, 2]
Y = [ 2, 1]

x = [X[0]+np.random.randn(nDots)*blur, X[1]+np.random.randn(nDots)*blur]
y = [Y[0]+np.random.randn(nDots)*blur, Y[1]+np.random.randn(nDots)*blur]

np_data = np.hstack((x,y)).T

data = torch.tensor(np_data).float()

labels = torch.zeros(len(data),1)

labels[100:200] = 1

ANN1 = nn.Sequential(
    nn.Linear(2,100),
    nn.ReLU(),
    nn.Linear(100,100),
    nn.ReLU(),
    nn.Linear(100,1),
    nn.Sigmoid()
)

numepochs = 100
losses = torch.zeros(numepochs)

lossfun = nn.MSELoss()

optimization = torch.optim.SGD(ANN1.parameters(), lr=0.1)

for epoch in range(numepochs):

    yHat = ANN1(data)

    loss = lossfun(yHat, labels)
    losses[epoch] = loss

    optimization.zero_grad()
    loss.backward()
    optimization.step()

predictions = (yHat > 0.5).float()

Acc = 100*torch.mean((predictions == labels).float())
print(Acc)






