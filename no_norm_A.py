import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img, plot_dict
from plt_env import *
from soln_analysis import SolnAnalysis

lsc_path = f'results/bars_lsc/no_norm_A'
def plot_lsc_A_norm(out=False):
    plt.figure(figsize = (8,4))
    analysis = SolnAnalysis(lsc_path)
    plt.ylabel(r'Mean Dict. Norm')
    norm_A = np.linalg.norm(analysis.A, axis=1)
    for n, norm in enumerate(norm_A.T[norm_A[-1].argsort()]):
        q = n / len(norm_A.T)
        plt.plot(analysis.time, norm, c='k', alpha=.1)
    plt.xlabel('Time')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.tight_layout()
    if out:
        plt.savefig('./figures/lsc_A_norm.pdf', bb_inches='tight')
    plt.show()

def plot_lsc_dict(out=False):
    analysis = SolnAnalysis(lsc_path)
    A = analysis.A[-1]
    plt.figure(figsize=(8, 8))
    norm_A = np.linalg.norm(analysis.A, axis=1)
    plot_dict(A[:, (-norm_A[-1]).argsort()], (8, 8), 8, 8)
    #plt.tight_layout()
    plt.subplots_adjust(hspace=.1, wspace=.1, left=.05, right=.95, bottom=.05, top=.95)
    if out:
        plt.savefig('./figures/lsc_dict.pdf', bb_inches='tight')
    plt.show()


#plot_lsc_dict(True)
plot_lsc_A_norm(True)
assert False

plt.subplot(212)
plt.title('DSC and LCA')
plt.ylabel(r'Mean Dict. Norm')
dir_path = f'results/bars_lca/no_norm_A'
analysis = SolnAnalysis(dir_path)
norm_A = np.linalg.norm(analysis.A, axis=1)
norm_med = np.quantile(norm_A, q=0.5, axis=1)
norm_low = np.quantile(norm_A, q=0.2, axis=1)
norm_high = np.quantile(norm_A, q=0.8, axis=1)
plt.plot(analysis.time, norm_med, 'k-.', label='LCA')
plt.fill_between(analysis.time, norm_low, norm_high, color='grey', alpha=0.4)

plt.ylabel(r'Mean Dict. Norm')
dir_path = f'results/bars_dsc/no_norm_A'
analysis = SolnAnalysis(dir_path)
norm_A = np.linalg.norm(analysis.A, axis=1)
norm_med = np.quantile(norm_A, q=0.5, axis=1)
norm_low = np.quantile(norm_A, q=0.2, axis=1)
norm_high = np.quantile(norm_A, q=0.8, axis=1)
plt.plot(analysis.time, norm_med, 'k:', label='DSC')
plt.fill_between(analysis.time, norm_low, norm_high, color='grey', alpha=0.4)
plt.xlabel('Time / Iterations')
plt.legend()
plt.tight_layout()
#plt.savefig('figures/dict_norm.pdf', bb_inches='tight')
plt.show()
