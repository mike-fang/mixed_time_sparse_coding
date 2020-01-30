import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img, plot_dict
from plt_env import *
from soln_analysis import SolnAnalysis

plt.figure(figsize = (8,6))
plt.subplot(211)
plt.title('LSC')
plt.xticks([])
plt.ylabel(r'Mean Dict. Norm')
dir_path = f'results/bars_lsc/no_norm_A'
analysis = SolnAnalysis(dir_path)
norm_A = np.linalg.norm(analysis.A, axis=1)
norm_med = np.quantile(norm_A, q=0.5, axis=1)
norm_low = np.quantile(norm_A, q=0.2, axis=1)
norm_high = np.quantile(norm_A, q=0.8, axis=1)
plt.plot(analysis.time, norm_med, 'k', label='Median Dict. Norm')
plt.fill_between(analysis.time, norm_low, norm_high, color='grey', alpha=0.4, label='20%-80% range')
plt.legend()

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
plt.savefig('figures/dict_norm.pdf', bb_inches='tight')
plt.show()
