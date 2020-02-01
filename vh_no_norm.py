import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img, plot_dict
from plt_env import *
from soln_analysis import SolnAnalysis

def plot_dicts(out=False, sort=True):
    A = analysis.A[-1]
    norm = -np.linalg.norm(A, axis=0)
    plot_dict(A[:, norm.argsort()], (8, 8), 8, 16)
    print(A.shape)
    plt.suptitle('Sorted Learned Dictionary (128/128)')
    if out:
        plt.savefig('./figures/vh_unnormed_dict.pdf', bb_inches='tight')
    plt.show()

#dir_path = f'results/vh_oc_2_dim_8_lsc/no_norm_A'
dir_path = get_timestamped_dir(load=True, base_dir='vh_learn_pi_oc_2')
analysis = SolnAnalysis(dir_path)


plot_dicts(out=False)

norm_A = np.linalg.norm(analysis.A, axis=1)

for norm in norm_A.T:
    plt.plot(analysis.time, norm, c='k', alpha=.2, lw=.1)


plt.show()
