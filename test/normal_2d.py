import numpy as np
import sdeint
import tqdm
import matplotlib.pylab as plt

tspan = np.linspace(0, 100, int(1e4))
x0 = np.zeros(2)
tau = 1e-1

rho = 0.5
sigma = np.array([[1, rho], [rho, 1]]) * 100
sigma_inv = np.linalg.inv(sigma)

def f(x, t):
    return -(sigma_inv @ x + 1*np.sign(x))/tau

def g(x, t):
    n = len(x)
    return np.eye(n)/(tau**0.5)

y = sdeint.itoint(f, g, x0, tspan)
plt.subplot(211)
plt.scatter(*y.T, s=.1)
plt.subplot(212)
plt.hist(y[:, 0], bins=100)
plt.show()
assert False
tspan = tspan[::10]
y = y[::10]
plt.plot(y)
plt.subplot(212)
plt.hist(y, bins=100, density=True)
xspan = np.linspace(-3*sigma, 3*sigma, 1001)
f = (2 * np.pi * sigma**2)**(-0.5) * np.exp(-0.5 * xspan**2/sigma**2)
plt.plot(xspan, f)
print(f.sum() * (xspan[1] - xspan[0]))
plt.show()
