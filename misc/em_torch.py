import numpy as np
import sdeint
import torch as th
import matplotlib.pylab as plt

def f(x, t):
    return -x

def g(x, t):
    return 1


tspan = np.linspace(0, 1000, 10000)
y = sdeint.itoEuler(f, g, 0, tspan)
#plt.plot(tspan, y)
plt.hist()
plt.show()
