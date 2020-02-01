import torch as th
import numpy as np
from ctsc import *
from loaders import VanHaterenSampler
from visualization import show_img_XRA
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis

DIM = 8
OC = 2
BATCH_FRAC = 2
H = W = DIM
N_DIM = H * W
N_BATCH = int(N_DIM * BATCH_FRAC)
N_DICT = int(OC * N_DIM)
PI = 0.3
LOAD = False

N_S = 400
# DSC params
dsc_params = dict(
    n_A = 2500,
    n_s = N_S,
    eta_A = 0.2,
    eta_s = 0.05,
)

PI = round(float(PI), 2)
print(PI)
model_params = dict(
        n_dict=N_DICT,
        n_dim=N_DIM,
        n_batch=N_BATCH,
        positive=True,
        pi=PI,
        l1=1,
        sigma=1,
        )

base_dir = f'vh_learn_pi_oc_{OC}'
loader = VanHaterenSampler(H, W, N_BATCH)

solver_params = dsc_solver_param(**dsc_params)
solver_params['spike_coupling'] = False
solver_params['asynch'] = False
solver_params['T_u'] = 1
solver_params['tau_u0'] = 1e7

# Define model, solver
model = CTSCModel(**model_params)

solver = CTSCSolver(model, **solver_params)
dir_path = solver.get_dir_path(base_dir)
soln = solver.solve(loader, soln_T=N_S, soln_offset=-1, normalize_A=False)
solver.save_soln(soln)

assert False
X = soln['x'][:]
R = soln['r'][:]
A = soln['A'][:]

out_path = os.path.join(dir_path, 'evol.mp4')
out_path = None
show_img_XRA(X, R, A, n_frames=1e2, img_shape=(DIM, DIM), out_file=out_path)

