import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader, ZIELoader
from visualization import show_img_XRA, show_batch_img, plot_dict
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis
from glob import glob

BASE_DIR = 'learn_pi_test'

N_DIM = 1
N_BATCH = 1000
N_DICT = 1
PI = 1
SIGMA = .5
L1 = 1
#loader = BarsLoader(8, 8, 13, p=PI, sigma=SIGMA, l1=0.5)
A_list = {}
for SIGMA in [2, 1, .5, .2, .1]:
    try:
        f_name = f'./results/learn_pi_test_sigma_{SIGMA}.npy'
        A = np.load(f_name)
    except:
        loader = ZIELoader((1, 1), N_BATCH, pi=PI, l1=L1, sigma=SIGMA)

        N_A = 500
        N_S = 500
        ETA_A = 0.01
        #ETA_A = 1e-50
        ETA_S = 0.01
        # model params
        model_params = dict(
                n_dict=N_DICT,
                n_dim=N_DIM,
                n_batch=N_BATCH,
                positive=True,
                pi=PI,
                l1=L1,
                sigma=SIGMA,
                )

        model = CTSCModel(**model_params)
        model.A.data = .5 * model.A.abs()

        solver_params = CTSCSolver.get_dsc(model, n_A=N_A, n_s=N_S, eta_A=ETA_A, eta_s=ETA_S, return_params=True)
        solver_params['spike_coupling'] = False
        solver_params['asynch'] = False
        solver_params['T_u'] = 1
        #solver_params['mu_u'] = .1
        #solver_params['mu_A'] = .01
        #solver_params['tau_u0'] = 1e7

        solver = CTSCSolver(model, **solver_params)
        dir_path = solver.get_dir_path(BASE_DIR, name=None, overwrite=True)
        soln = solver.solve(loader, soln_T=N_S, out_energy=False, normalize_A=False, norm_A_init=.1, report_norm_pi=True)
        solver.save_soln(soln)
        f_name = f'./results/learn_pi_test_sigma_{SIGMA}.npy'
        A = soln['A'].flatten()
        np.save(f_name, A)
    A_list[SIGMA] = A

for s in A_list:
    A = A_list[s]
    u = s / 2
    c = (u, 0.5 * u * (1-u), 1-u)
    plt.plot(A, label=fr'$\sigma = {s}$')

plt.plot([0, len(A)], [1, 1], 'g--', label='True Norm')
plt.legend()
plt.show()
