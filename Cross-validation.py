import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader,TensorDataset

import seaborn as sns
iris = sns.load_dataset('iris')

data = torch.tensor( iris[iris.columns[0:4]].values ).float()

labels = torch.zeros(len(data), dtype=torch.long)

labels[iris.species=='versicolor'] = 1
labels[iris.species=='virginica'] = 2

#Fake dataset
fakedata = np.tile(np.array([1,2,3,4]),(10,1)) \
      + np.tile(10*np.arange(1,11),(4,1)).T
fakelabels = np.arange(10)>4

fakedataLdr = DataLoader(fakedata, shuffle=True)

fakeDataset = TensorDataset( torch.Tensor(fakedata), torch.Tensor(fakelabels) )

fakedataLdr = DataLoader(fakeDataset, shuffle=True)

#Split the data
train_data, test_data, train_labels, test_labels = \
train_test_split(fakedata,fakelabels, test_size=.2)

#Convert them into Pythorch Datasets
train_data = TensorDataset(
    torch.Tensor(train_data),torch.Tensor(train_labels)
)

test_data = TensorDataset(
    torch.Tensor(test_data),torch.Tensor(test_labels)
)

#translate into dataloader objects
train_loader = DataLoader(train_data,batch_size=4)
test_loader = DataLoader(test_data)

#real data

#split the data
train_data,test_data, train_labels,test_labels = \
train_test_split(data, labels, train_size=.8)

#convert to PyTorch Dataset
train_data = TensorDataset(
    train_data,train_labels
)

test_data = TensorDataset(
    test_data,test_labels
)

#translate to dataloader objects
train_loader = DataLoader(train_data,shuffle=True,batch_size=12)
test_loader = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

def createANewModel():
    ANNiris = nn.Sequential(
        nn.Linear(4,64),
        nn.ReLU(),
        nn.Linear(64,64),
        nn.ReLU(),
        nn.Linear(64,3),
        )

    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ANNiris.parameters(), lr=.01)

    return ANNiris,lossfun,optimizer

numepochs = 500

def trainTheModel():
    ANNiris,lossfun,optimizer = createANewModel()

    trainAcc = []
    testAcc = []

    for epochi in range(numepochs):

        batchAcc = []
        for X,y in train_loader:

            yHat = ANNiris(X)
            loss = lossfun(yHat,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchAcc.append( 100*torch.mean(
                (torch.argmax(yHat,axis=1) == y).float()).item()
                )

        trainAcc.append( np.mean(batchAcc) )

        X,y = next(iter(test_loader))
        predlabels = torch.argmax( ANNiris(X),axis=1 )
        testAcc.append( 100*torch.mean((predlabels == y).float()).item() )

    return trainAcc,testAcc


ANNiris,lossfun,optimizer = createANewModel()

trainAcc,testAcc = trainTheModel()

fig = plt.figure(figsize=(10,5))

plt.plot(trainAcc,'ro-')
plt.plot(testAcc,'bs-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend(['Train','Test'])
plt.show()
