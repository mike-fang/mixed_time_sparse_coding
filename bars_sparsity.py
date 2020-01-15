from soln_analysis import SolnAnalysis
import numpy as np
from ctsc import get_timestamped_dir
import matplotlib.pylab as plt
import seaborn as sns

def plot_density(analysis, name, color, thresh=2e-2):
    analysis.zero_coeffs(thresh)
    mean_nz = analysis.mean_nz()
    time = analysis.time
    plt.plot(time, mean_nz, 'k', color=color, label=name)

def plot_nz_distr():
    colors = ['b', 'g', 'r', 'k']
    exps = ['lsc', 'ctsc', 'asynch', 'dsc']
    #fig = plt.figure(figsize=(8, 3))
    for c, exp in zip(colors, exps):
        base_dir = f'bars_{exp}'
        dir_path = get_timestamped_dir(load=True, base_dir=base_dir)
        print(dir_path)
        #dir_path = get_timestamped_dir(load=True, base_dir=base_dir, dir_name='bars_complete')
        analysis = SolnAnalysis(dir_path)
        analysis.zero_coeffs(thresh=2e-2)
        analysis.plot_nz_hist(last_frac=.2, s_max=6, title=None, n_bins=100, log=False, eps_s=1e-5, ylim='auto')

        plt.savefig(f'./figures/bars_distr_{exp}.pdf')
        plt.show()

if __name__ == '__main__':
    colors = ['k', 'g', 'r', 'b']
    exps = ['dsc', 'ctsc', 'asynch', 'lsc']
    fig = plt.figure(figsize=(8, 3))
    for c, exp in zip(colors, exps):
        base_dir = f'bars_{exp}'
        dir_path = get_timestamped_dir(load=True, base_dir=base_dir)
        analysis = SolnAnalysis(dir_path)
        plot_density(analysis, name=exp, color=c, thresh=1e-2)
    time = analysis.time
    plt.plot(time, np.ones_like(time) * 0.3, 'm--', label=r'$\pi = 0.3$')
    plt.xlim(0, 1e4)
    plt.legend()
    plt.ylabel('Density')
    plt.xlabel('Time')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'./figures/bars_density.pdf')
    plt.show()
