import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm

N, M = 10, 5
EPS = 1e-10
C = 1e-5
tau = C*0.5

A = np.random.uniform(size=(N, M))
G = A.T @ A
Q_ij = np.zeros_like(G)
I = np.random.uniform(size = M)
V = np.ones_like(I) * EPS

tspan = np.linspace(0, 1000*tau, 10000)

t0 = 0
for t in tspan:
    dt = t - t0
    V += EPS
    _I_ij = G * V[None, :]
    I_ij = _I_ij * (I/(G@V))[:, None]
    dQ_ij = (I_ij - _I_ij)
    Q_ij += dQ_ij * dt
    V = (M*C)**(-1) * Q_ij.sum(axis=0)
    print(dQ_ij.max())
    t0 = t
    #print(I_ij.sum())
