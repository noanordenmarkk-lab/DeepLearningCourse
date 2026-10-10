import pandas as pd
import numpy as np

class model1(object):

    def __init__(self,numlayers, numunits, name):
        self.layers = numlayers # attributes
        self.units = numunits
        self.name = name
        self.weights = np.random.rand(numlayers, self.units)

    def trainModel(self, x, y):
        return x * self.weights + y

m1 = model1(1, 5, "X")
print(m1.weights)

    
