import h5py
import numpy as np
from ctsc import get_timestamped_dir
import matplotlib.pylab as plt
import os.path
import seaborn as sns

def plot_mse(exp, color, n_bins=100, start_frac=0, end_frac=1, alpha=.3, q=0.2):
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
        mse_low.append(np.quantile(mse[inter], q))
        mse_high.append(np.quantile(mse[inter], 1-q))

    plt.plot(T0, -np.log(mse_med), color=color, label=exp)
    plt.fill_between(T0, -np.log(mse_high), -np.log(mse_low), color=color, alpha=alpha)

def plot_energy(exp, color):
    base_dir = f'bars_{exp}'
    dir_path = get_timestamped_dir(load=True, base_dir=base_dir)
    soln = h5py.File(os.path.join(dir_path, 'soln.h5'))
    print(soln['A'])
    t = soln['t'][:]
    A = soln['A'][:]
    A = A.reshape((-1, 8, 8, 16))
    A0 = A[-1].copy()
    A0[A0 < .1] = 0
    A0[A0 > .1] = 8**(-0.5)

    cos_sim = np.sum(A * A0, axis=(1, 2))
    mean_cs = cos_sim.mean(axis=1)
    plt.plot(t, mean_cs, color=color, label=exp)
    plt.ylim(0, 1)


colors = {'dsc': 'black', 'ctsc': 'g', 'asynch': 'r', 'lsc': 'blue'}

if True:
    Q = .2
    fig = plt.figure(figsize=(8, 3))
    for exp in colors:
        c = colors[exp]
        plot_mse(exp, c, q=Q, n_bins=50, start_frac=0, end_frac=1, alpha=.2)
    plt.legend(loc=4)
    plt.ylabel('Reconstr. NL-MSE')
    plt.xlabel('Time')
    plt.xlim(0, 1e5)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('./figures/bars_mse.pdf')
    plt.show()
else:

    fig = plt.figure(figsize=(8, 3))
    for exp in colors:
        c = colors[exp]
        plot_energy(exp, c)
    plt.ylabel('Dict. Cosine Similarity')
    plt.xlabel('Time')
    plt.xlim(0, 1e5)
    plt.legend(loc=4)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('./figures/bars_cosine.pdf')
    plt.show()
