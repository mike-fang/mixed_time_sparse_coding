from euler_maruyama import *
import torch as th
import numpy as np
import matplotlib.pylab as plt
from torch.nn import Parameter
from matplotlib import animation

GAMMA = 2.5
def p_cauchy(x):
    return (np.pi * GAMMA * (1 + (x/GAMMA)**2))**(-1)
def E_cauchy(x):
    return th.log(1 + (x/GAMMA)**2)

steps = int(5e4)

x = Parameter(th.tensor(0.))

param_groups = [
        {'params': [x], 'tau': 1, 'mu': 0, 'T': 1},
        ]

solver = EulerMaruyama(param_groups, dt=1)

X = []
def closure():
    energy = E_cauchy(x)
    energy.backward()
    return energy

for _ in range(steps):
    solver.zero_grad()
    solver.step(closure)
    X.append(float(x))

x = th.linspace(-15, 15, 100)
p = p_cauchy(x)

fig, axes = plt.subplots(nrows=2)

N_FRAMES = 200
N_SKIP = int(steps//N_FRAMES)
def animate(nf):
    N = nf * N_SKIP
    _X = X[:N]
    for ax in axes:
        ax.clear()
    axes[0].plot(_X, 'k')
    axes[1].hist(_X, bins=x, density=True, fc='black')
    axes[1].plot(x.data.numpy(), p.data.numpy())

anim = animation.FuncAnimation(fig, animate, frames=N_FRAMES-1, interval=100, repeat=True)
anim.save('./cauchy_no_mass.mp4')
plt.show()

