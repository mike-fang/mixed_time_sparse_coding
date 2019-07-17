import numpy as np
import torch as th
import matplotlib.pylab as plt

s0=1
s = np.linspace(-5, 5, 50)
u = (s - s0*(np.sign(s))) * (np.abs(s) > s0)
h = (np.abs(s) - s0) * (np.abs(s) < s0)
plt.plot(s, h)
plt.show()
