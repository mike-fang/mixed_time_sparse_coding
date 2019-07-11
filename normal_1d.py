import numpy as np
import sdeint
import tqdm
import matplotlib.pylab as plt

tspan = np.linspace(0, 100, int(1e4))
x0 = 0
tau = 1e-2
sigma = 1e-0

def f(x, t):
    return -x/(tau * sigma**2)

def g(x, t):
    return  np.ones_like(x)/(tau**0.5)

y = sdeint.itoint(f, g, x0, tspan)
tspan = tspan[::10]
y = y[::10]
plt.subplot(211)
plt.plot(y)
plt.subplot(212)
plt.hist(y, bins=100, density=True)
xspan = np.linspace(-3*sigma, 3*sigma, 1001)
f = (2 * np.pi * sigma**2)**(-0.5) * np.exp(-0.5 * xspan**2/sigma**2)
plt.plot(xspan, f)
print(f.sum() * (xspan[1] - xspan[0]))
plt.show()
