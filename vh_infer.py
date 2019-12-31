import torch as th
import numpy as np
from ctsc import *
from loaders import VanHaterenSampler
from visualization import show_img_XRA
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis
from vh_patches import vh_loader_model_solver

# DSC params
dsc_params = dict(
    n_A = 250/1e9,
    n_s = 100*1e9,
    eta_A = 0.1,
    eta_s = 0.1,
)
DIM = 8
OC = 1

PI = 0.3

EXP = '1T'

dict_dir_path = os.path.join('./results/vh_1T/exp_250_100')

loader, model_params, solver_params = vh_loader_model_solver(dim=DIM, batch_frac=0.5, dict_oc=OC, dsc_params=dsc_params, pi=PI, exp=EXP)

# Define model, solver
model = CTSCModel(**model_params)
solver = CTSCSolver(model, **solver_params)

#solver.load_checkpoint(dir_path=dict_dir_path)
out_dir = solver.get_dir_path('vh_rand_infer')

soln = solver.solve(loader, out_N=1e4, save_N=1)
solver.save_soln(soln)

X = soln['x'][:]
R = soln['r'][:]
A = soln['A'][:]

out_path = os.path.join(out_dir, 'evol.mp4')
show_img_XRA(X, R, A, n_frames=1e2, img_shape=(DIM, DIM))
