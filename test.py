import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

n = 100

D = np.zeros((n, 2))

D[:, 0] = np.linspace(0,10, n)+np.random.randn(n)
D[:, 1] = D[:, 0]**2 + np.random.randn(n)*10

df = pd.DataFrame({
    'x': D[:, 0],
    'y': D[:, 1]
})

sns.regplot(x='x', y='y', data=df)
plt.show()