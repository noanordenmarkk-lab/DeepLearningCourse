import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython import display
plt.rcParams['figure.dpi'] = 150

def modelX(N,x,y):
#build model
    ANNreg = nn.Sequential(
    nn.Linear(1,1),
    nn.ReLU(),
    nn.Linear(1,1)
    )
    #params
    learningRate = .05
    lossfun = nn.MSELoss()
    optimizer = torch.optim.SGD(ANNreg.parameters(),lr=learningRate)

    #train model
    numepochs = 500
    losses = torch.zeros(numepochs)
    for epochi in range(numepochs):
        #forward pass
        yHat = ANNreg(x)

        #losses
        loss = lossfun(yHat,y)
        losses[epochi] = loss

        #Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predictions = ANNreg(x)
    testloss = (predictions-y).pow(2).mean()

    return predictions, losses

def CreateData(m):
    N = 50
    x = torch.randn(N,1)
    y = m*x + torch.randn(N,1)/2

    return(N,x,y)


m = np.linspace(-2,2,21)

results = np.zeros((len(m),50,2))

for i in range(len(m)):
    index = 0
    while index < 50:
        N,x,y = CreateData(m[i])
        predictions,losses = modelX(N,x,y)

        results[i,index,0] = losses[-1].item()
        results[i,index,1] = np.corrcoef(
            y.T,
            predictions.detach().T
            )[0,1]
        index +=1
results[np.isnan(results)] = 0

fig,ax = plt.subplots(1,2,figsize=(12,4))

ax[0].plot(m,np.mean(results[:,:,0],axis=1),'ko-',markerfacecolor='w',markersize=10)
ax[0].set_xlabel('Slope')
ax[0].set_ylabel('Loss')

ax[1].plot(m,np.mean(results[:,:,1],axis=1),'ms-',markerfacecolor='w',markersize=10)
ax[1].set_xlabel('Slope')
ax[1].set_ylabel('Real-predicted correlation')
ax[1].set_title('Medel performance')

plt.show()


    



