import numpy as np
import torch

import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset,DataLoader

import matplotlib.pyplot as plt

data = np.loadtxt(open('Datasets/mnist_train_small.csv','rb'),delimiter=',')

labels = data[:8,0]
data   = data[:8,1:]

dataNorm = data/np.max(data)

dataNorm = dataNorm.reshape(dataNorm.shape[0],1,28,28)

dataT   = torch.tensor( dataNorm ).float()
labelsT = torch.tensor( labels ).long()

class customDataset(Dataset):
    def __init__(self, tensors, transform=None):

        assert all(tensors[0].size(0)==t.size(0) for t in tensors), 'Size mismatch between tensors'

        self.tensors   = tensors
        self.transform = transform

    def __getitem__(self, index):

        if self.transform:
            x = self.transform(self.tensors[0][index])
        else:
            x = self.tensors[0][index]

        y = self.tensors[1][index]

        return x,y

    def __len__(self):
        return self.tensors[0].size(0)

#create a list of transforms to apply to the image
imgtrans = T.Compose([
    T.ToPILImage(),
    T.RandomVerticalFlip(p=.5),
    #T.RandomRotation(90),
    T.ToTensor()
])

train_data = customDataset((dataT,labelsT),imgtrans)

dataLoaded = DataLoader(train_data,batch_size=8,shuffle=False)

# import data from the dataloader, just like during training
X,y = next(iter(dataLoaded))


# create a figure
fig,axs = plt.subplots(2,8,figsize=(16,4))


# loop over images in the dataset
for i in range(8):

  # draw images
  axs[0,i].imshow(dataT[i,0,:,:].detach(),cmap='gray')
  axs[1,i].imshow(X[i,0,:,:].detach(),cmap='gray')

  # some niceties
  for row in range(2):
    axs[row,i].set_xticks([])
    axs[row,i].set_yticks([])

# row labels
axs[0,0].set_ylabel('Original')
axs[1,0].set_ylabel('torch dataset')

plt.show()