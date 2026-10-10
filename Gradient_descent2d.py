import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

from IPython import display

# def peaks(x, y):
#     x, y = np.meshgrid(x, y)

#     z = 3*(1-x)**2 * np.exp(-(x**2) - (y+1)**2) \
#     - 10*(x/5 - x**3 - y**5) * np.exp(-x**2-y**2) \
#     - 1/3*np.exp(-(x+1)**2 - y**2)

#     return z

# x = np.linspace(-3, 3, 201)
# y = np.linspace(-3, 3, 201)

# Z = peaks(x, y)

# plt.imshow(Z, extent=[x[0],x[-1],y[0],y[-1]],vmin=-5,origin='lower')

# sx,sy = sym.symbols('sx,sy')

# sZ = 3*(1-sx)**2 * sym.exp(-(sx**2) - (sy+1)**2) \
#     - 10*(sx/5 - sx**3 - sy**5) * sym.exp(-sx**2-sy**2) \
#     - sym.Rational(1, 3)*sym.exp(-(sx+1)**2 - sy**2)

# df_x = sym.lambdify( (sx, sy),sym.diff(sZ,sx), 'sympy')
# df_y = sym.lambdify( (sx, sy),sym.diff(sZ,sy), 'sympy')

# localMin = np.random.rand(2)*4-2
# startpoint = localMin[:]

# learning_rate = 0.1
# training_epochs = 100

# trajectory = np.zeros((training_epochs, 2))
# for i in range(training_epochs):
#     grad = np.array([ df_x(localMin[0], localMin[1]),
#                       df_y(localMin[0], localMin[1])
#                       ])
#     localMin = localMin - learning_rate*grad
#     trajectory[i,:] = localMin

# plt.imshow(Z, extent=[x[0], x[-1], y[0], y[-1]], vmin=-5, vmax=5, origin='lower')
# plt.plot(startpoint[0], startpoint[1], 'bs')
# plt.plot(localMin[0], localMin[1], 'ro')
# plt.plot(trajectory[:, 0], trajectory[:, 1], 'r')
# plt.legend(['rnd start', 'local min'])
# plt.colorbar()
# plt.show()

x = np.linspace(-2*np.pi,2*np.pi,401)

def f(x):
    return np.sin(x) * np.exp(-x**2*.5)

def deriv(x):
    return np.exp(-x**2 * 0.5) * (np.cos(x) - x * np.sin(x))

startlocs = np.linspace(-5,5,50)
finalres = np.zeros(len(startlocs))

learning_rate = 0.01
training_epochs = 1000

localmin = np.random.choice(x,1)

for idx,localmin in enumerate(startlocs):
    for i in range(training_epochs):
        learning_rate = learning_rate * (deriv(i)**2)
        grad = deriv(localmin)
        localmin = localmin - learning_rate*grad

    finalres[idx] = localmin

print(learning_rate)

plt.plot(startlocs,finalres,'s-')
plt.xlabel('Starting Guess')
plt.ylabel('Final Guess')
plt.show()



