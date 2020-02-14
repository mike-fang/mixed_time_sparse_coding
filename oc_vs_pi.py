import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img, plot_dict, animate_dict
from plt_env import *
from soln_analysis import SolnAnalysis
from matplotlib import animation
from tqdm import tqdm


oc_list = [1, 1.5, 2, 2.5, 3, 3.5, 4, 6]
pi_mean = []
pi_err = []
pi_low = []
pi_high = []
exp = 'learn_pi'
dim = 8
#oc_list = [0.5, 1, 2, 3, 4, 1.5, 2.5, 3.5]
#oc_list.sort()
try:
    pi_mean = np.load('./results/pi_mean.npy')
    pi_high = np.load('./results/pi_high.npy')
    pi_low = np.load('./results/pi_low.npy')
except:
    pi_mean = []
    pi_high = []
    pi_low = []
    for oc in tqdm(oc_list):
        base_dir = f'vh_{exp}_dim_{dim}_oc_{oc}'.replace('.', 'p')
        print(base_dir)
        try:
            d = get_timestamped_dir(load = True, base_dir=base_dir)
            analysis = SolnAnalysis(d)
        except:
            d = get_timestamped_dir(load = True, base_dir=base_dir, index=-2)
            analysis = SolnAnalysis(d)
        pi = analysis.pi
        pi = pi[-len(pi)//10:]

        pi_mean.append(pi.mean())
        pi_high.append(np.quantile(pi, .9))
        pi_low.append(np.quantile(pi, .1))

    np.save('./results/pi_mean.npy', pi_mean)
    np.save('./results/pi_high.npy', pi_high)
    np.save('./results/pi_low.npy', pi_low)

oc_list = np.array(oc_list)
pi_err = np.vstack((pi_high - pi_mean, pi_mean - pi_low))
num_active = pi_mean * (64 * oc_list)

mean_active = num_active[:-2].mean()
plt.errorbar(oc_list, num_active, pi_err * (64 * oc_list), fmt='ok')
plt.plot([1, 6], [mean_active]*2, 'r--', label=fr'Mean Num. Active $\approx {mean_active:.1f}$')
plt.xlabel('Overcompleteness')
plt.ylabel('Mean Num. of Dict. Elements Active')
plt.legend()
plt.ylim(10, 30)
plt.savefig( f'figures/mean_activity_exp_{exp}.pdf')
plt.show()
oc_arr = np.linspace(1, 6, 100)
pi_arr = mean_active / oc_arr / 64
plt.errorbar(oc_list, pi_mean, pi_err, fmt='ok')
plt.plot(oc_arr, pi_arr, 'r--', label=rf'$\pi \propto {mean_active:.1f}/\Omega$ Fit')
plt.xlabel(r'Overcompleteness ($\Omega$)')
plt.ylabel(r'Mean Actvity ($\pi$)')
plt.legend()
plt.savefig( f'figures/oc_vs_pi_exp_{exp}.pdf')
plt.show()
