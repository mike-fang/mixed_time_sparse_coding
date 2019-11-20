import sys
sys.path.append('..')
from euler_maruyama import EulerMaruyama
import numpy as np
import torch as th
from torch.nn import Parameter
import matplotlib.pylab as plt
from tqdm import tqdm

N_DICT = 2
N_DIM = 2
DT_SCALE = 1.5
TAU_SCALE = 1000
OUTMAX = 1000000000

A = th.FloatTensor(N_DIM, N_DICT).normal_()
A = th.tensor([[5, 5], [-1, 1.]])

def p(s, sigma):
    return np.exp(-0.5 * s**2 / sigma**2) / (2 * np.pi * sigma**2)**0.5
        
def E(s):
    return 0.5 * ((A @ s)**2).sum()

s = Parameter(th.FloatTensor(N_DICT).normal_())
param_groups = [
        {'params': [s], 'tau': 1, 'mu': 0, 'T': 1},
        ]

U, Sigma, V = th.svd(A)
M = (Sigma.max())**2
m = (Sigma.min())**2
dt = DT_SCALE/M

tmax = int(TAU_SCALE/m/dt)

solver = EulerMaruyama(param_groups, dt=dt)
S = []
def closure():
    energy = E(s)
    energy.backward()
    return energy

S = np.zeros((tmax, N_DIM))
for n in tqdm(range(tmax)):
    solver.zero_grad()
    solver.step(closure)
    S[n] = s.data.numpy()

if tmax > OUTMAX:
    skip = int(tmax//OUTMAX)
    S = S[::skip]


fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
V = np.array(V)
Sigma = np.array(1/Sigma)
S_ = S@V
axes[0].scatter(*S.T, c='k', s=1)
axes[0].plot([0, Sigma[0] * V[0, 0]], [0, Sigma[0] * V[1, 0]], 'r')
axes[0].plot([0, Sigma[1] * V[0, 1]], [0, Sigma[1] * V[1, 1]], 'g')
axes[0].set_aspect(1)

s_ = np.linspace(-3*Sigma[0], 3*Sigma[0])
p_ = p(s_, Sigma[0])
axes[1].hist(S_[:, 0], fc='grey', bins=100, density=True)
axes[1].plot(s_, p_, 'r--')

s_ = np.linspace(-3*Sigma[1], 3*Sigma[1])
p_ = p(s_, Sigma[1])
axes[2].hist(S_[:, 1], fc='grey', bins=100, density=True)
axes[2].plot(s_, p_, 'g--')
plt.suptitle(rf'$dt = {DT_SCALE:.2f} M^{{-1}}$ ; $T = {TAU_SCALE} \mu^{{-1}}$')
plt.savefig(f'../figures/convergence/{TAU_SCALE}_{DT_SCALE}.pdf')
plt.show()
