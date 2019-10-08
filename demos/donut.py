import sys
sys.path.append('..')
from euler_maruyama import EulerMaruyama
import torch as th
from torch.nn import Parameter
import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm

steps = 1000
noise_cache = 0

tau_x = -1
tau_t = 100
tau_mu = 30
SIMGA = 1
MU0 = 0
dt = 1
scale = int(100/dt)
x = Parameter(th.tensor(1.))
x0 = Parameter(th.tensor(0.))
t = Parameter(th.tensor(-3.))
mu = Parameter(th.tensor(3.))
print(th.exp(-t/2))
def energy(x, t, mu):
    return 0.5 * th.exp(t) * (x-mu)**2
def next_batch():
    x0.data.normal_(0, SIMGA)
def closure():
    E_eff = 0
    E_eff += energy(x0, t, mu) 
    E_eff -= energy(x, t, mu)
    E_eff.backward()
    return E_eff


param_groups = [
        {'params': [x], 'tau': tau_x, 'T': 1.000},
        {'params': [t], 'tau': tau_t, 'mu':.5, 'T': 0},
        {'params': [mu], 'tau': tau_mu, 'mu':.5, 'T': 0},
        ]
solver = EulerMaruyama(param_groups, dt=.01)

X = np.zeros(steps)
X0 = np.zeros(steps)
T = np.zeros(steps)
MU0 = np.zeros(steps)
MU = np.zeros(steps)
for n in tqdm(range(steps)):
    if n % (abs(tau_x)*scale) == 0:
        next_batch()
    solver.zero_grad()
    solver.step(closure)
    X[n] = float(x)
    X0[n] = float(x0)
    T[n] = float(t)
    MU[n] = float(mu)

N_FRAMES = 10
N_SKIP = int(steps//N_FRAMES)

nf = 3
N = nf * N_SKIP
N0 = max(0, (nf-5) * N_SKIP)

fig, axes = plt.subplots(nrows=2)

axes[0].plot(np.ones(steps)[N0:N]*SIMGA, 'b--')
axes[0].plot(-np.ones(steps)[N0:N]*SIMGA, 'b--')
axes[0].plot(X0[N0:N], 'k', label = 'X ~ data')
axes[0].plot(X[N0:N], 'r', label = 'X ~ model')
axes[0].plot(MU[N0:N] + np.exp(-T[N0:N]/2), 'g--', label = 'Scale')
axes[0].plot(MU[N0:N] -np.exp(-T[N0:N]/2), 'g--', label = 'Scale')
axes[0].plot(MU[N0:N], 'g-', label = 'Mean')
plt.legend()
plt.show()
