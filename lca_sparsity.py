from soln_analysis import SolnAnalysis
import numpy as np
from ctsc import get_timestamped_dir
import matplotlib.pylab as plt

dir_path = get_timestamped_dir(load=True, base_dir='bars_lca')
analysis = SolnAnalysis(dir_path, lca=True)
mean_nz = analysis.mean_nz()

plt.plot(mean_nz)
plt.show()
