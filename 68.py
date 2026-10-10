import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

N = 100
x = torch.randn(N,1)
y = x + torch.randn(N,1)

ANNreg = nn.Sequential(
    nn.Linear(1,1),
    nn.ReLU(),
    nn.Linear(1,1)
)

lr = 0.05
lossfun = nn.MSELoss()
optimizer = torch.optim.SGD(ANNreg.parameters(), lr=lr)

trainingSetSize = ((N*0.8)/N)

x_train,x_test, y_train, y_test = \
train_test_split(x,y, train_size=trainingSetSize)

numepochs = 500
for epochi in range(numepochs):
    yHat = ANNreg(x_train)

    loss = lossfun(yHat,y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

predYtest = ANNreg(x_test)
testloss = (
    predYtest-y_test
    ).pow(2).mean()

predYtrain = ANNreg(x_train).detach().numpy()

plt.plot(x,y,'k^',label='All data')
plt.plot(x_train, predYtrain,
         'bs',markerfacecolor='w',label='Training pred.')
plt.plot(x_test, predYtest.detach(),
         'ro',markerfacecolor='w',label='Test pred.')
plt.legend()
plt.show()