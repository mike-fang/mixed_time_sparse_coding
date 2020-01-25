from soln_analysis import SolnAnalysis
import numpy as np
from ctsc import get_timestamped_dir
import matplotlib.pylab as plt
exp_colors = {'dsc':'k', 'lca':'g', 'lsc':'b', 'lsc_0.1':'m'}
exp_names = {'dsc':'DSC', 'lca':'LCA', 'lsc':'LSC', 'lsc_0.1': r'LSC; $\pi=0.1$'}
def plot_s_dkl(out=False):
    for exp in exp_colors:
        color = exp_colors[exp]
        if '0.1' in exp:
            dir_path = f'./results/bars_lsc/wrong_pi_2'
        else:
            dir_path = get_timestamped_dir(load=True, base_dir=f'bars_{exp}')

        analysis = SolnAnalysis(dir_path)
        analysis.zero_coeffs()
        time, dkls = analysis.dkls_history(t_bins=50, n_bins=50, s_max=6)
        if 'lsc' in exp:
            time /= 2

        plt.plot(time, dkls, c=color, label=exp_names[exp])
        plt.ylabel(r'$D_{KL}$ estimate of $s_i$')
        plt.xlabel(r'Time')
        plt.ylim(-0.01, .3)
    plt.legend(loc=1)
    if out:
        plt.savefig('./figures/bars_dkl.pdf')
    plt.show()
def plot_activity(out=False):
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
    if out:
        plt.savefig(f'./figures/bars_activity.pdf')
    plt.ylim(0, .6)
    plt.show()

for exp in exp_colors:
    if '0.1' in exp:
        continue
    color = exp_colors[exp]
    name = exp_names[exp]
    dir_path = get_timestamped_dir(load=True, base_dir=f'bars_{exp}')
    analysis = SolnAnalysis(dir_path)
    time, dklx = analysis.dklx_history(t_bins=50, n_bins=50)
    if exp == 'lsc':
        time /= 2
    plt.plot(time, dklx, c=color, label=name)
plt.legend()
plt.show()
