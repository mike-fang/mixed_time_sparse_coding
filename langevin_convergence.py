import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img, plot_dict
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis

DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

DICT = 'learned'
EXP = 'lsc'

# Define loader
H = W = 8
N_DIM = H * W
OC = 1
N_BATCH = 2 * (H * W)
N_DICT = OC * (H + W)
PI = 0.01
SIGMA = 1e-20
L1 = 1.0
loader = BarsLoader(H, W, N_BATCH)

N_S = 400
N_A = 10
ETA_A = 1e-30
ETA_S = 0.02


# model params
model_params = dict(
        n_dict=N_DICT,
        n_dim=N_DIM,
        n_batch=N_BATCH,
        positive=True,
        pi=PI if EXP == 'lsc' else 1,
        l1=L1,
        sigma=SIGMA,
        )
base_dir = f'convergence_test'

# Define model, solver
model = CTSCModel(**model_params)
model = model.to(DEVICE)
if DICT == 'learned':
    model.A.data = loader.bases.t().to(DEVICE)
solver_params = CTSCSolver.get_dsc(model, n_A=N_A, n_s=N_S, eta_A=ETA_A, eta_s=ETA_S, return_params=True)
solver_params['spike_coupling'] = False
solver_params['asynch'] = False
solver_params['T_u'] = 1

for tau_u0 in [.5, 1, 2, 5, 10, 20, 50]:
    solver_params['tau_u0'] = tau_u0
    name = f'tau_u0_{tau_u0:.0e}'.replace('+', 'p').replace('-', 'm')

    solver = CTSCSolver(model, **solver_params)
    dir_path = solver.get_dir_path(base_dir, name=name, overwrite=True)
    soln = solver.solve(loader, soln_T=N_S, soln_offset=-1, out_energy=False)
    solver.save_soln(soln)
    continue
    X = soln['x'][:]
    R = soln['r'][:]
    A = soln['A'][:]

    plot_dict(A[-1], (8, 8), int(OC * 2), 8)

    out_path = None
    #out_path = os.path.join(dir_path, 'evol.mp4')
    #show_img_XRA(X, R, A, out_file=out_path, n_frames=1e2, img_shape=(H, W))

