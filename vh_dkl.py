from soln_analysis import SolnAnalysis
import numpy as np
from ctsc import get_timestamped_dir
from glob import glob
from plt_env import *
from collections import defaultdict
from tqdm import tqdm

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

dirs = glob('./results/vh_dim_8_lsc/l1_*')
pi_list = np.round(np.arange(0.05, 0.5, 0.05), 2)
l1_list = np.round(np.arange(0.5, 3, 0.5), 2)

try:
    dklx_arrs = np.load('./results/dklx_arr.npy')
    dkls_arrs = np.load('./results/dkls_arr.npy')
    assert False
except:
    dklx_arrs = np.zeros((len(l1_list), len(pi_list)))
    dkls_arrs = np.zeros((len(l1_list), len(pi_list)))

    for d in tqdm(dirs):
        analysis = SolnAnalysis(d)
        l1 = analysis.l1
        pi = analysis.pi
        l1_idx = np.where(l1_list == l1)[0][0]
        pi_idx = np.where(pi_list == pi)[0][0]

        time, dklx = analysis.dklx_history(t_bins=10, n_bins=50, s_max=5)
        time, dkls = analysis.dkls_history(t_bins=10, n_bins=50, s_max=5)

        if False:
            plt.plot(dkls)
        if False:
            plt.subplot(121)
            analysis.plot_nz_hist(end=.1)
            plt.subplot(122)
            analysis.plot_nz_hist(start=.9)
            plt.title(f'pi: {pi}, l1:{l1}')
            plt.show()

        dklx_arrs[l1_idx, pi_idx] = dklx[-1]
        dkls_arrs[l1_idx, pi_idx] = dkls[-1]
    np.save('./results/dklx_arr.npy', dklx_arrs)
    np.save('./results/dkls_arr.npy', dkls_arrs)

plt.subplot(211)
plt.imshow(dkls_arrs)
ylim = plt.ylim()
xlim = plt.xlim()
plt.yticks(range(len(l1_list)))
plt.xticks(range(len(pi_list)))
plt.ylim(*ylim)
plt.xlim(*xlim)
ax = plt.gca()
ax.set_xticklabels(pi_list)
ax.set_yticklabels(l1_list)
plt.xlabel(r'$\pi$')
plt.ylabel(r'$\lambda$')
plt.subplot(212)
plt.imshow(dklx_arrs)
ylim = plt.ylim()
xlim = plt.xlim()
plt.yticks(range(len(l1_list)))
plt.xticks(range(len(pi_list)))
plt.ylim(*ylim)
plt.xlim(*xlim)
ax = plt.gca()
ax.set_xticklabels(pi_list)
ax.set_yticklabels(l1_list)
plt.xlabel(r'$\pi$')
plt.ylabel(r'$\lambda$')
plt.show()
plt.subplot(211)
plt.plot(dkls_arrs[1], label='s')
plt.subplot(212)
plt.plot(dklx_arrs[1], label='x')
plt.show()
