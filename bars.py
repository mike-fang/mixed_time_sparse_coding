import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis

# Define loader
H = W = 8
N_DIM = H * W
N_BATCH = 2 * (H * W)
N_DICT = H + W
PI = 0.3
SIGMA = .5
loader = BarsLoader(H, W, N_BATCH, p=PI, sigma=SIGMA)
LARGE = False
N_S = 100
L1 = 0.2

# DSC params
if not LARGE:
    N_A = 1000
    N_S = N_S
    ETA_A = 0.02
    ETA_S = 0.02
else:
    N_A = 10
    N_S = 250
    ETA_A = 5
    ETA_S = 0.1


# model params
model_params = dict(
        n_dict=N_DICT,
        n_dim=N_DIM,
        n_batch=N_BATCH,
        positive=True,
        pi=1.,
        l1=L1,
        sigma=SIGMA,
        )

EXP = 'lsc'
LOAD = False
assert EXP in ['dsc', 'ctsc', 'asynch', 'lsc']
if LARGE:
    base_dir = f'bars_large_step_{EXP}'
else:
    base_dir = f'bars_{EXP}'

# Define model, solver
model = CTSCModel(**model_params)
model.A.data = th.tensor(np.load('./A0.npy'))
solver_params = CTSCSolver.get_dsc(model, n_A=N_A, n_s=N_S, eta_A=ETA_A, eta_s=ETA_S, return_params=True)
if EXP == 'dsc':
    pass
elif EXP == 'ctsc':
    solver_params['spike_coupling'] = False
elif EXP == 'asynch':
    solver_params['spike_coupling'] = False
    solver_params['asynch'] = True
elif EXP == 'lsc':
    solver_params['spike_coupling'] = False
    solver_params['asynch'] = True
    solver_params['T_u'] = 1
    model.pi = PI

# Load or make soln
if LOAD:
    dir_path = get_timestamped_dir(load=True, base_dir=base_dir)
    soln = h5py.File(os.path.join(dir_path, 'soln.h5'))
else:
    solver = CTSCSolver(model, **solver_params)
    #dir_path = solver.get_dir_path(base_dir)
    solver.get_dir_path(base_dir)
    soln = solver.solve(loader, soln_T=N_S, soln_offset=-1)
    #soln = solver.solve(loader, soln_T=N_S * 5 - 1)
    solver.save_soln(soln)

X = soln['x'][:]
R = soln['r'][:]
A = soln['A'][:]

show_img_XRA(X, R, A, n_frames=1e2, img_shape=(H, W))
plt.show()
