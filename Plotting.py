import matplotlib.pyplot as plt
import numpy as np
cmap = plt.cm.viridis

fig, ax = plt.subplots(1, figsize=(10, 5))

x = [0.0, 10.0]
y = [0.0, 0.0]

for i in range(20):
    x += [0.0, 10.0]
    y += [i-(1.15*i), i+(0.001*i)]
    ax.plot(x[2*i:2*i+2], y[2*i:2*i+2], color=cmap(i / 19), linewidth=2)

ax.set_xlim(0, 10)
ax.set_ylim(-10, 10)

plt.show()