import numpy as np
import matplotlib.pylab as plt

u = np.linspace(-4, 4, 401)

s = np.zeros_like(u)
u_ = u[np.abs(u) > 1]
s[np.abs(u) > 1] = u_ - np.sign(u_)

plt.plot(u, s, 'k')
ax = plt.gca()
ax.set_aspect(1)
ax.grid(True, which='both')
plt.xlabel(r'$u$', fontsize=12)
plt.ylabel(r'$s = f(u)$', fontsize=12)
plt.savefig('./figures/threshold_fn.pdf')
plt.show()
