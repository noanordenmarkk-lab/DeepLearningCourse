import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

nPerClus = 100
blur = 1

A = [ 1, 3 ]
B = [ 1, -2]

a = [A[0]+np.random.randn(nPerClus)*blur, A[1]+np.random.randn(nPerClus)*blur]
b = [B[0]+np.random.randn(nPerClus)*blur, B[1]+np.random.randn(nPerClus)*blur]

labels_np = np.vstack((np.zeros((nPerClus,1)),np.ones((nPerClus,1))))

data_np = np.hstack((a,b)).T

data = torch.tensor((data_np)).float()
labels = torch.tensor((labels_np)).float()

class ANNm(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(2,16)

        self.layer = nn.Linear(16,1)

        self.output = nn.Linear(1,1)

    def forward(self,x):

        x = self.input(x)
        x = F.relu(x)
        x = self.layer(x)
        x = F.relu(x)
        x = self.output(x)
        x = torch.sigmoid(x)

        return x

ANNmulti = ANNm()

LR = .05

lossfun = nn.BCELoss()

optimizer = torch.optim.SGD(ANNmulti.parameters(), lr = LR)

numepochs = 1000
losses = np.zeros(numepochs)

for epochi in range(numepochs):

    yHat = ANNmulti(data)

    loss = lossfun(yHat, labels)
    losses[epochi] = loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

predictions = ANNmulti(data)

totalacc = 100*torch.mean(((predictions>.5) == labels).float())

print('Final accuracy: %g%%' %totalacc)


# show the losses
plt.plot(losses,'o',markerfacecolor='w',linewidth=.1)
plt.title('Final accuracy: %g%%' %totalacc)
plt.xlabel('Epoch'), plt.ylabel('Loss')
plt.show()

