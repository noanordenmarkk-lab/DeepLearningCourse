import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from imageio import imread
import IPython as ipy

n = 10

H = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        H[i, j]= 1/(i+j+1)

sns.heatmap(H, vmax=0.4)
plt.show()