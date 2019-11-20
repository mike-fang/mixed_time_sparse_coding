import sys
sys.path.append('..')
from euler_maruyama import EulerMaruyama
import numpy as np
import torch as th
from torch.nn import Parameter
import matplotlib.pylab as plt
from tqdm import tqdm

x = Parameter(
        th.tensor(1.)
        )

param_groups = [
        {'params': [x], 'tau': 1, 'mu': 0, 'T': 0},
        ]

solver = EulerMaruyama(param_groups, dt=0.01)

def closure():
    energy = 0.5 * x**2
    energy.backward()
    return energy

tmax = 1000
X = np.zeros((tmax))
for n in tqdm(range(tmax)):
    solver.zero_grad()
    solver.step(closure)
    X[n] = x.data.numpy()
plt.plot(X)
plt.show()
