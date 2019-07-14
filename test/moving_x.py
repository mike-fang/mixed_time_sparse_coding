import numpy as np
import sdeint
import tqdm
import matplotlib.pylab as plt

tspan = np.linspace(0, 100, int(1e4))
s0 = np.zeros(2)
sigma = 2
tau_s = 1e-1
tau_x = 10
sparsity = 1

rho = 0.0
theta = np.pi/8
A = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)],
    ])
print(A)
X = np.array([
    [-1 , -1],
    [-1 , +1],
    [+1 , +1],
    [+1 , -1]
    ])
X *= 10


def get_x_idx(t):
    return ((t//tau_x) % len(X)).astype(int)
def f(s, t):
    x_idx = get_x_idx(t)
    x = X[x_idx]

    dH_recontr = -A.T @ (x.T - A@s) / sigma**2
    dH_sparse = sparsity * np.sign(s)
    return -(dH_recontr + dH_sparse) / tau_s
def g(s, t):
    n = len(s)
    return np.eye(n)/(tau_s**0.5)

y = sdeint.itoint(f, g, s0, tspan)

fig, ax = plt.subplots()

x_idx = get_x_idx(tspan)
x = X[x_idx]
ax.scatter(*x.T, c=x_idx, cmap='Set1')
print(y.shape)
ax.scatter(*(A @ y.T), s=1, c=x_idx, cmap='Set1')

ax.set_aspect(1)
plt.show()
