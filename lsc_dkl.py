from soln_analysis import SolnAnalysis
import numpy as np
from ctsc import get_timestamped_dir
import matplotlib.pylab as plt

plt.figure(figsize=(8, 4))
exp_colors = {'dsc':'k', 'lca':'g', 'lsc':'b'}
if True:
    for exp in exp_colors:
        color = exp_colors[exp]
        dir_path = get_timestamped_dir(load=True, base_dir=f'bars_{exp}')

        analysis = SolnAnalysis(dir_path)
        analysis.zero_coeffs()
        time, dkls = analysis.dkls_history(t_bins=50, n_bins=50, s_max=6)
        if exp == 'lsc':
            time /= 2
        plt.plot(time, dkls, c=color, label=exp)
        plt.ylabel(r'$D_{KL}$ estimate of $s_i$')
        plt.ylim(-0.01, .3)
    plt.legend(loc=1)
    plt.savefig('./figures/bars_dkl.pdf')
    plt.show()

if False:
    for exp in exp_colors:
        color = exp_colors[exp]
        dir_path = get_timestamped_dir(load=True, base_dir=f'bars_{exp}')
        analysis = SolnAnalysis(dir_path)
        analysis.zero_coeffs()
        time, mnz = analysis.binned_mean_nz(t_bins=200)
        plt.ylim(0, 1)
        plt.ylabel('Mean Activity')
        plt.xlabel('Time')
        if exp == 'lsc':
            time /= 2
        plt.plot(time, mnz, c=color, label=exp)
    plt.plot((0, 0.5*analysis.time.max()), (.3, .3), 'r--', label=r'$\pi = 0.3$')
    plt.legend(loc=1)
    plt.savefig(f'./figures/bars_activity.pdf')
    plt.ylim(0, .6)
    plt.show()
