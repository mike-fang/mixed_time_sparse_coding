import matplotlib.pylab as plt
import numpy as np
import h5py
from glob import glob
import os.path
from soln_analysis import SolnAnalysis

pi_dirs = glob('./results/bars_lsc/PI_*')

p_list = []
pis_med = []
pis_low = []
pis_high = []
c_list = []
for d in pi_dirs:
    p = float(d.split('_')[-1].replace('p', '.'))
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

    pis = (s > 0).mean(axis=(1, 2))
    pi_med = np.median(pis)
    pi_high = np.quantile(pis, 0.9)
    pi_low = np.quantile(pis, 0.1)

    p_list.append(p)
    pis_med.append(pi_med)
    pis_high.append(pi_high)
    pis_low.append(pi_low)

c_list = np.array(c_list)
pi_list = np.array(pis_med)
p_list = np.array(p_list)
where_3 = c_list == 'b'
where_1 = c_list == 'r'
pis_high = np.array(pis_high)
pis_low = np.array(pis_low)
errs = np.vstack((pis_high-pis_med, pis_med-pis_low))

plt.errorbar(p_list[where_3], pi_list[where_3], yerr=errs[:, where_3], fmt='o', c='b', label=r'Data Activity: $\pi = 0.3$')
plt.errorbar(p_list[where_1], pi_list[where_1], yerr=errs[:, where_1], fmt='o', c='r', label=r'Data Activity: $\pi = 0.1$')
plt.plot((p_list.min(), p_list.max()), (.3, .3), 'b--')
plt.plot((p_list.min(), p_list.max()), (.1, .1), 'r--')
plt.xlabel(r'$\pi$')
plt.ylabel(r'Inference Activity')
plt.legend()
plt.ylim(0.00, .45)
plt.savefig('./figures/bars_lsc_pi.pdf', bb_inches='tight')


plt.show()


