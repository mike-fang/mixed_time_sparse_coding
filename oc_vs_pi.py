import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img, plot_dict, animate_dict
from plt_env import *
from soln_analysis import SolnAnalysis
from matplotlib import animation


oc_list = [1, 1.5, 2, 2.5, 3, 3.5, 4]
pi_list = []
pi_err = []
for oc in oc_list:
    base_dir = f'vh_learn_pi_dim_8_oc_{oc}'.replace('.', 'p')
    d = get_timestamped_dir(load = True, base_dir=base_dir)
    analysis = SolnAnalysis(d)
    pi = analysis.pi
    pi = pi[-len(pi)//10:]
    pi_list.append(pi.mean())
    pi_err.append(np.std(pi))

oc_arr = np.linspace(1, 4, 100)
pi_arr = pi_list[0] / oc_arr
plt.errorbar(oc_list, pi_list, pi_err, color='k')
plt.plot(oc_arr, pi_arr, 'r--')
plt.xlabel('Overcompleteness')
plt.ylabel('Mean Actvity')
plt.savefig('figures/oc_vs_pi.pdf')
plt.show()
