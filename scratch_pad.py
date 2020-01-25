import numpy as np
import matplotlib.pylab as plt

N = 1000
D = 2
sigma = 1
X = np.random.normal(0, sigma, size=(N, D))

s_max = 5
n_bins = 50
X_idx = np.round(X * n_bins / (s_max * sigma), 0)
X_idx += n_bins//2
X_idx[X_idx < 0] = 0
X_idx[X_idx >= n_bins] = n_bins - 1



plt.scatter(*X_idx.T)
plt.show()
