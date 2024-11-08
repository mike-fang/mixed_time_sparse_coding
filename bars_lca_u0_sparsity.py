import matplotlib.pylab as plt
import numpy as np
import h5py
from glob import glob
import os.path
from soln_analysis import SolnAnalysis

u0_dirs = glob('./results/bars_lca/pi_*')

u0_list = []
pi_list = []
err_list = []
c_list = []
for d in u0_dirs:
    u0 = float(d.split('_')[-1].replace('p', '.'))
    PI = float(d.split('_')[-3].replace('p', '.'))
    if PI == .3:
        c_list.append('b')
    elif PI == .1:
        c_list.append('r')

    if False:
        analysis = SolnAnalysis(d)
        analysis.zero_coeffs()
        s = analysis.S
    else:
        soln = h5py.File(os.path.join(d, 'soln.h5'))
        s = soln['s'][:]
    N, _, _ = s.shape

    pis = (s > 0).mean(axis=(1, 2))
    pi_med = np.median(pis)
    pi_high = np.quantile(pis, 0.9)
    pi_low = np.quantile(pis, 0.1)


    pi = (s[:] > 0).mean()
    u0_list.append(u0)
    pi_list.append(pi_med)
    err_list.append((pi_high - pi_med, pi_med - pi_low))

u0_list = np.array(u0_list)
pi_list = np.array(pi_list)
c_list = np.array(c_list)
err_list = np.array(err_list).T
where_3 = c_list == 'b'
where_1 = c_list == 'r'


plt.errorbar(u0_list[where_3], pi_list[where_3], err_list[:, where_3], c='b', fmt='o', label=r'LCA: $\pi = 0.3$')
plt.errorbar(u0_list[where_1], pi_list[where_1], err_list[:, where_1], c='r', fmt='o', label=r'LCA: $\pi = 0.1$')
#plt.scatter(u0_list[where_3], pi_list[where_3], c='b', label=r'Data Activity: $\pi = 0.3$')
#plt.scatter(u0_list[where_1], pi_list[where_1], c='r', label=r'Data Activity: $\pi = 0.1$')

plt.plot((u0_list.min(), u0_list.max()), (.3, .3), 'b--')
plt.plot((u0_list.min(), u0_list.max()), (.1, .1), 'r--')
plt.xlabel(r'$\lambda$')
plt.ylabel(r'Inference Activity')
plt.legend()
plt.ylim(0.00, .45)
#plt.savefig('./figures/bars_lca_u0.pdf', bb_inches='tight')

