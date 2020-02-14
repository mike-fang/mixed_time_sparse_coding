from soln_analysis import SolnAnalysis
import numpy as np
from ctsc import get_timestamped_dir
import matplotlib.pylab as plt
import matplotlib

font = {'family' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)
exp_colors = {'dsc':'k', 'lca':'g', 'lsc':'b', 'lsc_0.1':'m'}
exp_names = {'dsc':'DSC', 'lca':'LCA', 'lsc':'LSC', 'lsc_0.1': r'LSC; $\pi=0.1$'}
exp_colors = {'dsc':'k', 'lca':'g', 'lsc':'b'}
exp_names = {'dsc':'DSC', 'lca':'LCA', 'lsc':'LSC'}

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

        plt.plot(time, dkls, c=color, label=exp_names[exp])
        plt.ylabel(r'$D_{KL}$ estimate of $s_i$')
        plt.ylim(-0.01, .3)
    plt.legend(loc=1)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('Time')
    if out:
        plt.savefig('./figures/bars_dkl.pdf')
def plot_s_corr(out=False):
    for exp in exp_colors:
        color = exp_colors[exp]
        if '0.1' in exp:
            dir_path = f'./results/bars_lsc/wrong_pi_2'
        else:
            dir_path = get_timestamped_dir(load=True, base_dir=f'bars_{exp}')

        analysis = SolnAnalysis(dir_path)
        analysis.zero_coeffs()
        time, corr = analysis.corr_s_hist(t_bins=50)

        plt.plot(time, corr, c=color, label=exp_names[exp])
        plt.ylabel(r'$1 - \det(C)^{1/D}$')
    plt.legend(loc=1)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('Time')
    if out:
        plt.savefig('./figures/bars_corr_s.pdf')
def plot_activity(out=False):
    for exp in exp_colors:
        color = exp_colors[exp]
        dir_path = get_timestamped_dir(load=True, base_dir=f'bars_{exp}')
        analysis = SolnAnalysis(dir_path)
        analysis.zero_coeffs()
        time, mnz = analysis.binned_mean_nz(t_bins=200)
        plt.ylabel('Mean Activity')
        plt.plot(time, mnz, c=color, label=exp)
    plt.plot((0, analysis.time.max()), (.3, .3), 'r--', label=r'$\pi = 0.3$')
    plt.legend(loc=1, ncol=2)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('Time')
    plt.ylim(0, 1)
    if out:
        plt.savefig(f'./figures/bars_activity.pdf')
def plot_dkl_x(out=False):
    dklx_list = {}
    corr_list = {}
    for exp in exp_colors:
        if '0.1' in exp:
            continue
        color = exp_colors[exp]
        name = exp_names[exp]
        dir_path = get_timestamped_dir(load=True, base_dir=f'bars_{exp}')
        analysis = SolnAnalysis(dir_path)
        time, dklx = analysis.dklx_history(t_bins=50, n_bins=50)
        dklx_list[exp] = dklx
        time, corr = analysis.det_corr_hist(t_bins=50)
        corr_list[exp] = corr
    plt.subplot(211)
    for exp in dklx_list:
        plt.plot(time, dklx_list[exp], c=exp_colors[exp], label=exp_names[exp])
    plt.ylabel(r'$D_{KL}$ estimate of $\epsilon_i$')
    plt.xticks([])
    plt.ylim(0, 0.025)
    plt.legend()
    plt.subplot(212)
    for exp in corr_list:
        plt.plot(time, corr_list[exp], c=exp_colors[exp], label=exp_names[exp])
    plt.ylabel(r'$1 - \det(C)^{1/D}$')
    plt.xlabel('Time')
    plt.ylim(0, 0.15)
    plt.legend()
    if out:
        plt.savefig('./figures/bars_dklx.pdf')
    plt.show()

if __name__ == '__main__':
    plt.figure(figsize=(8, 3))
    plot_s_dkl(out=False)
    #plot_s_corr(out=True)
    plt.gcf().subplots_adjust(bottom=0.18)
    plt.show()
