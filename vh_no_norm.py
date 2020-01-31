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
dir_path = get_timestamped_dir(load=True, base_dir='vh_oc_2_dim_8_lsc')
analysis = SolnAnalysis(dir_path)


plot_dicts(out=True)

norm_A = np.linalg.norm(analysis.A, axis=1)

for q in np.linspace(0, 1, 6, endpoint=True)[::-1]:
    plt.plot(analysis.time, np.quantile(norm_A, q=q, axis=1), c=(q, 1-q, 1-q), label=np.round(q, 2))

#plt.fill_between(analysis.time, norm_low, norm_high, color='grey')
norm = norm_A[-1]
#plt.plot(norm[(-norm).argsort()])
plt.legend()
plt.savefig('./figures/vh_norm_A.pdf', bb_inches='tight')
plt.show()

