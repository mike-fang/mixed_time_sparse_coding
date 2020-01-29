import h5py
import numpy as np
from ctsc import get_timestamped_dir
import matplotlib.pylab as plt
import os.path
import seaborn as sns
from soln_analysis import SolnAnalysis
from glob import glob

exp = 'lsc'
base_dir = f'vh_dim_8_{exp}'
dir_path = get_timestamped_dir(load=True, base_dir=base_dir)
dir_paths = glob(os.path.join('results', 'vh_dim_8_lsc', '*'))
dir_paths.sort()

print(dir_paths)
assert False

for n, dir_path in enumerate(dir_paths):
    if n % 2 == 0:
        continue
    p = (n + 1) * 0.05
    p = round(p, 2)
    analysis = SolnAnalysis(dir_path)
    analysis.plot_nz_hist(last_frac=.5, s_max=4, n_bins=50, log=False)
    plt.savefig(f'./figures/lsc_diff_p/distr_{p}.pdf')

assert False
pasert
mean_mses = []
PI = []
plt.figure(figsize=(10, 6))
plt.subplot(211)
for n, dir_path in enumerate(dir_paths):
    p = (n + 1) * 0.05
    p = round(p, 2)
    PI.append(p)
    analysis = SolnAnalysis(dir_path)

    N_time = 100
    mse = np.median(analysis.mse(), axis=1)
    mse = mse.reshape(N_time, -1)
    mse_binned = np.median(mse, axis=1)
    mean_mses.append(np.mean(mse_binned[-50:]))
    plt.plot(mse_binned, label= f'{p:.2f}' if p in [.1, .2, .3, .4, .5] else None, c=(1-p, p, p))
plt.legend()
plt.ylabel('MSE')
plt.xlabel('Time (a. u.)')
plt.subplot(212)
plt.plot(PI, mean_mses, 'k')
plt.ylabel('MSE')
plt.xlabel('Density')
plt.savefig('./figures/vh_density_mse.pdf', bb_inches='tight')
plt.show()
