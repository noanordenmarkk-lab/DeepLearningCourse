import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader,TensorDataset

fakedata = np.tile(np.array([1,2,3,4]),(10,1)) + np.tile(10*np.arange(1,11),(4,1)).T
fakelabels = np.arange(10)>4

partitions = [.8,.1,.1]

#split data

#.8, .2
train_data,testTMP_data,train_labels,testTMP_labels = \
train_test_split(fakedata, fakelabels, train_size=partitions[0])

#.1, .1
split = partitions[1] / np.sum(partitions[1:])
devset_data,test_data, devset_labels, test_labels = \
train_test_split(testTMP_data, testTMP_labels, train_size=split)

batchsize = 16
train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)

print(train_loader.shape)


