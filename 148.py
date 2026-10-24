import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np
import scipy.stats as stats

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib_inline
import matplotlib
matplotlib.use("QtAgg")

def createNet():
  net = nn.Sequential(
    nn.Linear(2,8),
    nn.Linear(8,8),
    nn.Linear(8,1)
    )

  return net

  

net_noseed = createNet()
for p in net_noseed.named_parameters():
  if 'weight' in p[0]:
    nn.init.xavier_normal_(p[1].data)

net_rs1a = createNet()
for p in net_rs1a.named_parameters():
  if 'weight' in p[0]:
    torch.manual_seed(1)
    nn.init.xavier_normal_(p[1].data)

net_rs2 = createNet()
for p in net_rs2.named_parameters():
  if 'weight' in p[0]:
    torch.manual_seed(2)
    nn.init.xavier_normal_(p[1].data)

net_rs1b = createNet()
for p in net_rs1b.named_parameters():
  if 'weight' in p[0]:
    torch.manual_seed(1)
    nn.init.xavier_normal_(p[1].data)

def ProveTheWeights():
    fig,ax = plt.subplots(1,1,figsize=(17,7))

    ax.set_xlabel('Vectorized weight index')
    ax.set_ylabel('Weight value')
    ax.set_title('Prove The Weights!!!')

    weights0 = torch.cat([p.data.flatten() for p in net_noseed.parameters() if p.dim() > 1])
    weights1 = torch.cat([p.data.flatten() for p in net_rs1a.parameters() if p.dim() > 1])
    weights2 = torch.cat([p.data.flatten() for p in net_rs2.parameters() if p.dim() > 1])
    weights3 = torch.cat([p.data.flatten() for p in net_rs1b.parameters() if p.dim() > 1])

    weights = [weights0, weights1, weights2, weights3]
    markers = ['o', 's', '^', '+']
    colors  = ['r', 'k', 'm', 'g']
    names   = ['noseed', 'rs1a', 'rs2', 'rs1b']
    for i in range(4):
      ax.plot(weights[i],marker=markers[i],color=colors[i], label=names[i], linestyle='None')

    plt.legend()
    plt.show()

    

ProveTheWeights()