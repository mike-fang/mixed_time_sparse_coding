import torch as th
import numpy as np
from context import *

DICT = 'learned'
solver_type = 'lsc'

# Define loader
H = W = 8
N_DIM = H * W
OC = 1
N_BATCH = 20 * (H * W)
N_DICT = OC * (H + W)
PI = .3
SIGMA = .5
LARGE = False
L1 = 1.0
loader = BarsLoader(H, W, N_BATCH, p=PI, sigma=SIGMA, l1=L1)
NORM_A = False
NAME = 'no_norm_A'
NAME = None
if DICT == 'learned':
    NAME = 'learned_dict'
elif DICT == 'random':
    NAME = 'random_dict'

# model params
model_params = dict(
        n_dict=N_DICT,
        n_dim=N_DIM,
        n_batch=N_BATCH,
        positive=True,
        pi=PI if solver_type == 'lsc' else 1,
        l1=L1,
        sigma=SIGMA,
        )

# solver params
dsc_params = dict(
    n_s = 400,
    n_A = 400,
    eta_A = 0.05,
    eta_s = 0.05,
        )

base_dir = f'bars_{solver_type}'

def get_bars_solver(solver_type, model_params, dsc_params )
# Define model, solver
model = CTSCModel(**model_params)
model = model.to(DEVICE)
model.A.data *= 0.2
if DICT == 'learned':
    model.A.data = loader.bases.t().to(DEVICE)

solver_params = CTSCSolver.get_dsc(model, **dsc_params, return_params=True)

assert False
if solver_type == 'dsc':
    pass
elif solver_type == 'ctsc':
    solver_params['spike_coupling'] = False
elif solver_type == 'asynch':
    solver_params['spike_coupling'] = False
    solver_params['asynch'] = True
elif solver_type == 'lsc':
    solver_params['spike_coupling'] = False
    solver_params['asynch'] = False
    solver_params['T_u'] = 1

# Load or make soln
solver = CTSCSolver(model, **solver_params)
dir_path = solver.get_dir_path(base_dir, name=NAME, overwrite=True)
#solver.get_dir_path(base_dir)
soln = solver.solve(loader, soln_T=N_S, soln_offset=-1, out_energy=False, normalize_A=NORM_A, report_norm_pi=True)
solver.save_soln(soln)

X = soln['x'][:]
R = soln['r'][:]
A = soln['A'][:]

plot_dict(A[-1], (8, 8), int(OC * 2), 8)

out_path = os.path.join(dir_path, 'evol.mp4')
out_path = None
#show_img_XRA(X, R, A, out_file=out_path, n_frames=1e2, img_shape=(H, W))


