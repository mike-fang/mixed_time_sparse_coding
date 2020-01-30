from soln_analysis import SolnAnalysis
import numpy as np
from ctsc import get_timestamped_dir
from glob import glob
from plt_env import *
from collections import defaultdict
from tqdm import tqdm
from visualization import plot_dict

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
def get_dkl_arr(l1_list, pi_list):
    dklx_arrs = np.zeros((len(l1_list), len(pi_list)))
    dkls_arrs = np.zeros((len(l1_list), len(pi_list)))
    corr_arrs = np.zeros((len(l1_list), len(pi_list)))

    for d in tqdm(dirs):
        analysis = SolnAnalysis(d)
        l1 = analysis.l1
        pi = analysis.pi
        l1_idx = np.where(l1_list == l1)[0][0]
        pi_idx = np.where(pi_list == pi)[0][0]

        time, dklx = analysis.dklx_history(t_bins=10, n_bins=50, s_max=5)
        dklx = analysis.get_dkl_x(start=.9)
        dkls = analysis.get_dkl_s(start=.9)
        corr = analysis.det_corr_x(start=.9)
        #time, dkls = analysis.dkls_history(t_bins=10, n_bins=50, s_max=5)


        dklx_arrs[l1_idx, pi_idx] = dklx
        dkls_arrs[l1_idx, pi_idx] = dkls
        corr_arrs[l1_idx, pi_idx] = corr
    return dkls_arrs, dklx_arrs, corr_arrs
def plot_l1_pi_arr(arr):
    plt.imshow(arr)
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
def plot_dkl(dkls_arrs, dklx_arrs):
    plt.figure(figsize=(6, 8))
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
    plt.xlabel(r'$\pi$')
    plt.ylabel(r'$\lambda$')
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
def get_dir_name(idx, return_l1_pi=False):
    l1 = l1_list[idx[0]]
    pi = pi_list[idx[1]]
    if return_l1_pi:
        return f'results/vh_dim_8_lsc/l1_{l1}_p1_{pi}'.replace('.', 'p'), l1, pi
    else:
        return f'results/vh_dim_8_lsc/l1_{l1}_p1_{pi}'.replace('.', 'p')
def get_min_max_idx(arrs):
    max_idx = np.unravel_index(np.argmax(arrs), arrs.shape)
    min_idx = np.unravel_index(np.argmin(arrs), arrs.shape)
    return min_idx, max_idx

dirs = glob('./results/vh_oc_4_dim_8_lsc/l1_*')
l1_list = np.round(np.arange(0.5, 3, 0.5), 2)
pi_list = np.round(np.arange(0.05, 0.5, 0.05), 2)

try:
    dklx_arrs = np.load('./results/dklx_arr.npy')
    dkls_arrs = np.load('./results/dkls_arr.npy')
    corr_arrs = np.load('./results/corr_arr.npy')

except:
    dkls_arrs, dklx_arrs, corr_arrs = get_dkl_arr(l1_list, pi_list)
    np.save('./results/dkls_arr.npy', dkls_arrs)
    np.save('./results/dklx_arr.npy', dklx_arrs)
    np.save('./results/corr_arr.npy', corr_arrs)

min_s_idx, max_s_idx = get_min_max_idx(dkls_arrs)
min_x_idx, max_x_idx = get_min_max_idx(dklx_arrs)
min_c_idx, max_c_idx = get_min_max_idx(corr_arrs)

if True:
    plt.subplot(131)
    plt.title('s')
    plot_l1_pi_arr(dkls_arrs)
    plt.subplot(132)
    plt.title('x')
    plot_l1_pi_arr(dklx_arrs)
    plt.subplot(133)
    plt.title('c')
    plot_l1_pi_arr(corr_arrs)
    plt.show()
    assert False

d, l1, pi = get_dir_name(min_s_idx, return_l1_pi=True)
print(l1, pi)
analysis = SolnAnalysis(d)
A = analysis.A[-1]
plot_dict(A, (8, 8))
plt.gcf().suptitle(rf'$\lambda = {l1}; \pi = {pi}$')
plt.savefig('./figures/vh_min_dkls.pdf')
plt.show()

assert False
#time, corr = analysis.det_corr_hist(t_bins=100)
#plt.plot(time, corr)
analysis.plot_nz_hist(start=.9)
plt.show()
plt.subplot(211)
t, d = analysis.dkls_history(50, 30)
plt.plot(t, d)
plt.subplot(212)
t, c = analysis.corr_s_hist(50)
plt.plot(t, c)
plt.show()
