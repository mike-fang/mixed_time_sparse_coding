from soln_analysis import SolnAnalysis
import numpy as np
from ctsc import get_timestamped_dir
import matplotlib.pylab as plt

dir_path = get_timestamped_dir(load=True, base_dir='vh_dim_8_lca')
analysis = SolnAnalysis(dir_path, lca=True)

time = analysis.time
mse = np.mean(analysis.mse(), axis=1)
energy = analysis.energy()
mean_nz = analysis.mean_nz()

diff = analysis.diff()

D0 = (diff[:, 0, :])
D1 = (diff[:, 1, :])

plt.scatter(D0, D1)
plt.show()

assert False
plt.hist(diff.flatten(), bins=100)
print(np.std(diff.flatten()))
plt.xlim(-3, 3)
plt.yscale('log')
plt.savefig('./figures/lca_err_distr.pdf')
plt.show()
