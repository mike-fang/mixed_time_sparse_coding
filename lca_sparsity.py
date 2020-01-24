from soln_analysis import SolnAnalysis
import numpy as np
from ctsc import get_timestamped_dir
import matplotlib.pylab as plt

plt.figure(figsize=(8, 6))
plt.subplot(211)
dir_path = './results/bars_lca/random_dict'
analysis = SolnAnalysis(dir_path)
analysis.plot_nz_hist(start=0, end=1, s_max=4, n_bins=50, ylim=(0, 2))
plt.title('Random Dictionary')
plt.subplot(212)
dir_path = './results/bars_lca/trained_dict'
analysis = SolnAnalysis(dir_path)
analysis.plot_nz_hist(start=0, end=1, s_max=4, n_bins=50, ylim=(0,2))
plt.title('Trained Dictionary')
plt.subplots_adjust(hspace=.5)

plt.savefig(f'./figures/bars_distr_lca.pdf', bb_inches='tight')
plt.show()
