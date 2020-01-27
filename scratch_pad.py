import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm


N = int(1e5)
D = 2
sigma = 1
X = np.random.normal(0, sigma, size=(N, D))
Z = X / sigma

s_max = 3
n_bins = 5
bin_edges = np.linspace(-s_max, s_max, n_bins + 1)

X_idx = np.floor((Z + s_max) * n_bins / (2 * s_max)).astype(int)
X_idx[X_idx < 0] = -1
X_idx[X_idx > n_bins] = n_bins
X_idx += 1

X_idx, counts = np.unique(X_idx, return_counts=True, axis=0)

cdf = np.zeros(n_bins + 3)
cdf[1:-1] = norm.cdf(bin_edges)
cdf[-1] = 1
bin_prob = cdf[1:] - cdf[:-1]

p_i = np.prod(bin_prob[X_idx], axis=1)
plogq_i = p_i * np.log(counts)
Hpq = plogq_i.sum()
Hp = -D * (bin_prob * np.log(bin_prob)).sum()
print(Hp)
print(np.log(2 * N))
print(Hpq)
