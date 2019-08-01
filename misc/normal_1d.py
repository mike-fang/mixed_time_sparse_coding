import numpy as np
import sdeint
import tqdm
import matplotlib.pylab as plt
import torch as th
from time import time

tspan = np.linspace(0, 100, int(1e4))
x0 = 0
tau = 1e-2
sigma = 1e-0

def f(x, t):
    return -x/(tau * sigma**2)
def g(x, t):
    return  np.ones_like(x)/(tau**0.5)

def ito_euler(f, g, x0, tspan):
    # Convert to torch Tensor if not already
    if not isinstance(x0, th.Tensor):
        x0 = th.tensor(x0).float()
    if not isinstance(tspan, th.Tensor):
        tspan = th.tensor(tspan).float()

    # Get shape of x
    x_shape = list(x0.shape)
    if x_shape == []:
        n_dim = 1
    else:
        n_dim = x_shape[0]

    dt = tspan[1:] - tspan[:-1]
    dW = th.zeros((len(dt), n_dim))
    dW.normal_()
    dW *= dt[:, None]**0.5

    x = x0
    if n_dim == 1:
        y = th.zeros_like(tspan)
    else:
        y = th.zeros((len(tspan), n_dim))
    y[0] = x0
    for i, t in enumerate(tspan[1:]):
        a = th.tensor(f(x, t))
        b = th.tensor(g(x, t))
        if n_dim == 1:
            x = x + a * dt[i] + b * dW[i]
        else:
            x = x + a @ dt[i] + b @ dW[i]
        y[i] = x

    return y

y = sdeint.itoint(f, g, x0, tspan)
y_ = ito_euler(f, g, x0, tspan)
plt.subplot(211)
plt.plot(y)
plt.plot(y_.numpy())
plt.subplot(212)
plt.hist(y, bins=100, density=True)
plt.hist(y_.numpy().flatten(), bins=100, density=True)
xspan = np.linspace(-3*sigma, 3*sigma, 1001)
f = (2 * np.pi * sigma**2)**(-0.5) * np.exp(-0.5 * xspan**2/sigma**2)
plt.plot(xspan, f)
print(f.sum() * (xspan[1] - xspan[0]))
plt.show()
