import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

inChans  = 3 #RGB
outChans = 15
krnSize  = 5 # should be an odd number
stride   = 1
padding  = 0

c = nn.Conv2d(inChans,outChans,krnSize,stride,padding)

print(c)
print(' ')

print( 'Size of weights: ' + str(c.weight.shape) )
print( 'Sizd of bias: ' + str(c.bias.shape))