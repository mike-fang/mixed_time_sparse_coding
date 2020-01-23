import matplotlib.pylab as plt
import numpy as np
import h5py
from glob import glob
import os.path
from soln_analysis import SolnAnalysis

u0_dirs = glob('./results/bars_lca/u0_*')
print(u0_dirs)

u0_list = []
pi_list = []
for d in u0_dirs:
    u0 = float(d.split('_')[-1].replace('p', '.'))
    if False:
        analysis = SolnAnalysis(d)
        analysis.zero_coeffs()
        s = analysis.S
    else:
        soln = h5py.File(os.path.join(d, 'soln.h5'))
        s = soln['s'][:]
    N, _, _ = s.shape

    pi = (s[:-N//4] > 0).mean()
    u0_list.append(u0)
    pi_list.append(pi)

plt.scatter(u0_list, pi_list)

plt.show()