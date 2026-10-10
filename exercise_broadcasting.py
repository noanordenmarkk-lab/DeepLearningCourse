import numpy as np
import pandas as pd

v = np.array([0,1,2,3,4,5,6,7,8])

M = v.reshape(3,3)

C = np.tile(M, (3,1))

B = C*v

print(v)
print(M)
print(C)
print(C.shape)
print(B)