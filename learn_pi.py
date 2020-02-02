import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader, ZIELoader
from visualization import show_img_XRA, show_batch_img, plot_dict
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis

BASE_DIR = 'learn_pi_test'

N_DIM = 1
N_BATCH = 1000
N_DICT = 1
PI = 0.3
SIGMA = .5
L1 = 1
#loader = BarsLoader(8, 8, 13, p=PI, sigma=SIGMA, l1=0.5)
loader = ZIELoader((1, 1), N_BATCH, pi=PI, l1=L1, sigma=SIGMA)

N_A = 400
N_S = 1000
ETA_A = 0.1
#ETA_A = 1e-50
ETA_S = 5e-2

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
model.A.data[0] = th.tensor(1./L1)

solver_params = CTSCSolver.get_dsc(model, n_A=N_A, n_s=N_S, eta_A=ETA_A, eta_s=ETA_S, return_params=True)
solver_params['spike_coupling'] = False
solver_params['asynch'] = False
solver_params['T_u'] = 1
solver_params['tau_u0'] = 1e7

if True:
    solver = CTSCSolver(model, **solver_params)
    dir_path = solver.get_dir_path(BASE_DIR, name=None, overwrite=True)
    soln = solver.solve(loader, soln_T=N_S, soln_offset=-1, out_energy=False, normalize_A=False, norm_A_init=.1)
    solver.save_soln(soln)

dir_path = get_timestamped_dir(load=True, base_dir=BASE_DIR)
analysis = SolnAnalysis(dir_path)


analysis.plot_nz_hist(start=.9, s_max=20)
print((analysis.S > 8).mean())
plt.show()
analysis.plot_dict_norm()
plt.show()
plt.plot(analysis.pi)
plt.show()
assert False
plt.subplot(211)
plt.title(r'$\pi$')
plt.plot(soln['pi'])
plt.subplot(212)
plt.title(r'$||A||$')
plt.plot(np.linalg.norm(soln['A'], axis=(1,2)))
plt.show()