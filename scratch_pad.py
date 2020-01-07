import numpy as np
import matplotlib.pylab as plt


u = np.linspace(-2, 2, 100)

u0 = 1.5
def f(u):
    where_thresh = np.abs(u) <= u0
    s = np.zeros_like(u)
    s[~where_thresh] = u[~where_thresh] - u0 * np.sign(u[~where_thresh])
    return s

plt.plot(u, f(u))
plt.show()
