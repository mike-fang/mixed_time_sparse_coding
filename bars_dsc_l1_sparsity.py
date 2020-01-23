import matplotlib.pylab as plt
import numpy as np
import h5py
from glob import glob
import os.path
from soln_analysis import SolnAnalysis

l1_dirs = glob('./results/bars_dsc/l1_*')

try:
    l1_list = np.load('./results/dsc_sparsity/l1_list.npy')
    pi_list = np.load('./results/dsc_sparsity/pi_list.npy')
except:
    l1_list = []
    pi_list = []
    for d in l1_dirs:
        l1 = float(d.split('_')[-1].replace('p', '.'))
        if True:
            analysis = SolnAnalysis(d)
            analysis.zero_coeffs()
            s = analysis.S
        else:
            soln = h5py.File(os.path.join(d, 'soln.h5'))
            s = soln['s'][:]
        N, _, _ = s.shape

        if False:
            analysis.plot_nz_hist(last_frac=.25, s_max=4, n_bins=50, log=False)
            plt.show()
            plt.clf()
            plt.cla()
        pi = (s[:-N//8] > 0).mean()
        l1_list.append(l1)
        pi_list.append(pi)

    np.save('./results/dsc_sparsity/l1_list.npy', np.array(l1_list))
    np.save('./results/dsc_sparsity/pi_list.npy', np.array(pi_list))
plt.scatter(l1_list, pi_list, s=20, c='k')
plt.xlabel(r'$\lambda$')
plt.ylabel('Mean activity')
plt.plot((min(l1_list), max(l1_list)), (0.3, 0.3), 'r--', label='Data activity')
plt.legend()


plt.show()

