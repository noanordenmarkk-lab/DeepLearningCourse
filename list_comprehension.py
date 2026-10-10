import numpy as np
import pandas as pd
import time

#stop-start/num-1
step = (3+3)/10
x1 = [i for i in np.arange(-3,3,step)]
x2 = [i**2 for i in x1]

print("x1:", x1)
print("x2:", x2)
