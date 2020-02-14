import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img, plot_dict
from plt_env import *
from soln_analysis import SolnAnalysis
from glob import  glob

def plot_lsc_A_norm(out=False):
    plt.figure(figsize = (8,4))
    analysis.plot_dict_norm()
    plt.ylabel(r'Dict. Element Norm')
    plt.xlabel('Time')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.tight_layout()
    if out:
        plt.savefig('./figures/lsc_A_norm.pdf', bb_inches='tight')
    plt.show()
def plot_lsc_pi(out=False, pi=None):
    plt.figure(figsize = (8,4))
    plt.plot(analysis.time, analysis.pi, 'k', label='Model Activity')
    if pi is not None:
        plt.plot(analysis.time, [pi,] * len(analysis.time), 'r--', label='Data Activity')
    plt.ylabel(r'Activity ($\pi$)')
    plt.xlabel('Time')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.legend()
    if out:
        plt.savefig('./figures/lsc_pi', bb_inches='tight')
    plt.show()
def plot_lsc_dict(out=False):
    plt.figure(figsize=(8, 4))
    A = analysis.A[-1]
    plot_dict(A, (8, 8), 4, 8)
    if out:
        plt.savefig('./figures/lsc_dict.pdf', bb_inches='tight')
    plt.show()


#lsc_path = f'results/bars_lsc/no_norm_A'
#lsc_path = f'results/bars_learn_pi/no_norm_A'
#lsc_path = f'results/bars_learn_pi/fixed_pi'
#lspath = f'results/bars_learn_pi/learn_pi'
lsc_path = get_timestamped_dir(load=True, base_dir='bars_lsc')


if True:
    analysis = SolnAnalysis(lsc_path)
    plot_lsc_A_norm(False)
    #plot_lsc_pi(False, pi=0.3)
    plot_lsc_dict(False)


    assert False

#plt.subplot(212)
plt.figure(figsize=(8, 4))
plt.ylabel(r'Dict. Element Norm ')
analysis = SolnAnalysis(lsc_path)
norm_A = np.linalg.norm(analysis.A, axis=1)
norm_med = np.quantile(norm_A, q=0.5, axis=1)
norm_low = np.quantile(norm_A, q=0.2, axis=1)
norm_high = np.quantile(norm_A, q=0.8, axis=1)
plt.plot(analysis.time, norm_med, 'b-.', label='LSC')
plt.fill_between(analysis.time, norm_low, norm_high, color='blue', alpha=0.4)
#plt.plot([0, analysis.time[-1]], (1, 1), 'm--')

dir_path = f'results/bars_lca/no_norm_A'
analysis = SolnAnalysis(dir_path)
norm_A = np.linalg.norm(analysis.A, axis=1)
norm_med = np.quantile(norm_A, q=0.5, axis=1)
norm_low = np.quantile(norm_A, q=0.2, axis=1)
norm_high = np.quantile(norm_A, q=0.8, axis=1)
plt.plot(analysis.time /.4, norm_med, 'g-.', label='LCA')
plt.fill_between(analysis.time / .4, norm_low, norm_high, color='green', alpha=0.4)

dir_path = f'results/bars_dsc/no_norm_A'
analysis = SolnAnalysis(dir_path)
norm_A = np.linalg.norm(analysis.A, axis=1)
norm_med = np.quantile(norm_A, q=0.5, axis=1)
norm_low = np.quantile(norm_A, q=0.2, axis=1)
norm_high = np.quantile(norm_A, q=0.8, axis=1)
plt.plot(analysis.time, norm_med, 'r:', label='DSC')
plt.fill_between(analysis.time, norm_low, norm_high, color='red', alpha=0.4)
plt.fill_between([], [], [],color='grey', alpha=0.4, label='20%-80% range')
plt.xlabel('Time / Iterations')
plt.legend()
plt.tight_layout()
plt.savefig('figures/dict_norm_compare.pdf', bb_inches='tight')
plt.show()
