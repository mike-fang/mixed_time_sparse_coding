import numpy as np
import sdeint
import matplotlib.pylab as plt

tspan = np.linspace(0, 100, int(1e5))
l1 = 10 
x0 = 0

def f(x, t):
    return -l1*np.sign(x) * 1.

def g(x, t):
    return np.sqrt(2)
    return  np.ones_like(x)

y = sdeint.itoint(f, g, x0, tspan)

plt.subplot(211)
plt.plot(y)
plt.subplot(212)
plt.hist(y, bins=100, density=True)
xspan = np.linspace(-3/l1, 3/l1, 1001)
#l1 *= 2
f = 0.5 * np.exp(-l1*np.abs(xspan)) * l1
plt.plot(xspan, f)

plt.show()
