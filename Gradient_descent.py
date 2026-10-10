import numpy as np
import matplotlib.pyplot as plt
import torch

def fx(x):
   return 3*x**2 - 3*x + 4

def deriv(x):
   return 6*x - 3

x = np.linspace(-2, 2, 2001)

localminimum = np.random.choice(x, 1)

print("before:" + str(localminimum))

learning_rate = .01
training_epochs = 1000

for i in range(training_epochs):
   learning_rate = learning_rate * (1.1*.5)
   grad = deriv(localminimum)
   localminimum = localminimum - learning_rate*grad

print(localminimum)


