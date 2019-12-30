import torch as th
import numpy as np
from ctsc import *
from loaders import VanHaterenSampler
from visualization import show_img_XRA
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis

# Define loader
H = W = 8
N_DIM = H * W
N_BATCH = int(N_DIM // 2)
OC = 1.
N_DICT = int(OC * N_DIM)
PI = 0.3
loader = VanHaterenSampler(H, W, N_BATCH)

# DSC params
N_A = 100
N_S = 1000
ETA_A = 0.2
ETA_S = 0.1

# Model, solver params
model_params = dict(
        n_dict=N_DICT,
        n_dim=N_DIM,
        n_batch=N_BATCH,
        positive=True,
        pi=1.,
        l1=1,
        sigma=1.0,
        )

solver_params = dsc_solver_param(n_A=N_A, n_s=N_S, eta_A=ETA_A, eta_s=ETA_S)

EXP = 'asynch'
LOAD = False
assert EXP in ['dsc', 'ctsc', 'asynch', '1T']
base_dir = f'vh_{EXP}'

# Define model, solver
if EXP == 'dsc':
    pass
elif EXP == 'ctsc':
    solver_params['spike_coupling'] = False
elif EXP == 'asynch':
    solver_params['spike_coupling'] = False
    solver_params['asynch'] = True
elif EXP == '1T':
    solver_params['spike_coupling'] = False
    solver_params['asynch'] = True
    solver_params['T_u'] = 1
    model.pi = PI
model = CTSCModel(**model_params)
solver = CTSCSolver(model, **solver_params)

# Load or make soln
if LOAD:
    dir_path = get_timestamped_dir(load=True, base_dir=base_dir)
    soln = h5py.File(os.path.join(dir_path, 'soln.h5'))
else:
    solver = CTSCSolver(model, **solver_params)
    solver.get_dir_path(base_dir)
    soln = solver.solve(loader, out_N=1e4, save_N=1)
    solver.save_soln(soln)

X = soln['x'][:]
R = soln['r'][:]
A = soln['A'][:]

show_img_XRA(X, R, A, n_frames=1e2, img_shape=(H, W))

