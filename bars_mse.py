import h5py
import numpy as np
from ctsc import get_timestamped_dir
import matplotlib.pylab as plt
import os.path
import seaborn as sns

def plot_exp(exp, color, n_bins=100, start_frac=0, end_frac=1, alpha=.3):
    base_dir = f'bars_{exp}'
    dir_path = get_timestamped_dir(load=True, base_dir=base_dir)
    soln = h5py.File(os.path.join(dir_path, 'soln.h5'))
    t = soln['mse_t'][:]
    mse = soln['mse'][:]

    dt = int(t.max() // n_bins)

    t_inter = np.linspace(start_frac * t.max(), end_frac * t.max(), n_bins)
    T0 = t_inter[:-1]
    T1 = t_inter[1:]

    mse_med = []
    mse_high = []
    mse_low = []
    for t0, t1 in zip(T0, T1):
        inter = (t0 <= t) & (t < t1)
        mse_med.append(np.quantile(mse[inter], 0.5))
        mse_low.append(np.quantile(mse[inter], 0.2))
        mse_high.append(np.quantile(mse[inter], 0.8))




    plt.plot(T0, -np.log(mse_med), color=color, label=exp)
    plt.fill_between(T0, -np.log(mse_high), -np.log(mse_low), color=color, alpha=alpha)

START = .001
END = .3
plot_exp('dsc', 'b', start_frac=START, end_frac=END)
plot_exp('ctsc', 'g', start_frac=START, end_frac=END)
plot_exp('asynch', 'r', start_frac=START, end_frac=END, alpha=.2)
plot_exp('lsc', 'm', start_frac=START, end_frac=END, alpha=.2)
plt.legend()
plt.show()
