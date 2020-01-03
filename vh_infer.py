import torch as th
import numpy as np
from ctsc import *
from loaders import VanHaterenSampler, GaussainSampler
from visualization import show_img_XRA
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis
from vh_patches import vh_loader_model_solver

# DSC params
dsc_params = dict(
    n_A = 1000,
    n_s = 20,
    eta_A = 0.1,
    eta_s = 0.1,
)
DIM = 8
OC = 1

PI = 0.3

EXP = '1T'

dict_dir_path = os.path.join('./results/vh_1T/exp_250_100')

loader, model_params, solver_params = vh_loader_model_solver(dim=DIM, batch_frac=0.5, dict_oc=OC, dsc_params=dsc_params, pi=PI, exp=EXP)
#loader = GaussainSampler(H=DIM, W=DIM, n_batch=model_params['n_batch'])
solver_params['tau_A'] = 1e9
solver_params['T_u'] = 0
model_params['pi'] = 1
solver_params['asynch'] = False

print(solver_params)
assert False

# Define model, solver
model = CTSCModel(**model_params)
solver = CTSCSolver(model, **solver_params)

solver.load_checkpoint(dir_path=dict_dir_path)
out_dir = solver.get_dir_path('vh_fixed_dict_0T')
#out_dir = solver.get_dir_path('vh_noise')

soln = solver.solve(loader, out_N=1e3, save_N=1)
solver.save_soln(soln)

X = soln['x'][:]
R = soln['r'][:]
A = soln['A'][:]

out_path = os.path.join(out_dir, 'evol.mp4')
show_img_XRA(X, R, A, n_frames=1e2, out_file=out_path, img_shape=(DIM, DIM))
