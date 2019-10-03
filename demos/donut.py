import sys
sys.path.append('..')
from euler_maruyama import EulerMaruyama
import torch as th
from torch.nn import Parameter
import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm

steps = 10000
R = 2

x = Parameter(th.tensor(0.))
t = Parameter(th.tensor(1.))

param_groups = [
        {'params': [x], 'tau': 1, 'T': 1},
        {'params': [t], 'tau': 10, 'T': 1},
        ]
solver = EulerMaruyama(param_groups, dt=0.05)

def loader():
    return th.tensor(0.).uniform_()

def energy(x, t):
    return 0.5*t*x**2

def closure():
    energy = (t**2 + y**2) * (t**2 + y**2 - R**2)
    energy.backward()
    return energy

X = np.zeros(steps)
T = np.zeros(steps)
for n in tqdm(range(steps)):
    solver.zero_grad()
    solver.step(closure)
    X[n] = float(t)
    T[n] = float(y)

plt.scatter(X, Y, c='k', s=10)
plt.show()
