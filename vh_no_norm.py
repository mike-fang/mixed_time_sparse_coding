import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img, plot_dict
from plt_env import *
from soln_analysis import SolnAnalysis

dir_path = f'results/vh_oc_4_dim_8_lsc/no_norm_A'
analysis = SolnAnalysis(dir_path)
A = analysis.A[-1]
plot_dict(A, (8, 8), 10, 15)
plt.suptitle('Unormalized Learned Dictionary')
plt.savefig('./figures/vh_unnormed_dict.pdf', bb_inches='tight')
plt.show()

assert False
norm_A = np.linalg.norm(analysis.A, axis=1)
norm_med = np.quantile(norm_A, q=0.5, axis=1)
norm_low = np.quantile(norm_A, q=0.2, axis=1)
norm_high = np.quantile(norm_A, q=0.8, axis=1)

plt.plot(analysis.time, norm_med, 'k')
plt.fill_between(analysis.time, norm_low, norm_high, color='grey')
plt.tight_layout()
plt.savefig('./figures/vh_norm_A.pdf', bb_inches='tight')
plt.show()

