from loaders import BarsLoader
import matplotlib.pylab as plt
import numpy as np
import torch as th
from visualization import show_img_evo, show_img_XRA
from ctsc import *
#from L0_sparsity import *

H = W = 4
N_BATCH = H + W
N_DICT = H + W
PI = 0.3
loader = BarsLoader(H, W, N_BATCH, p=PI)

model_params = dict(
        n_dict = N_DICT,
        n_dim = H * W,
        n_batch = N_BATCH,
        positive=True,
        sigma=1,
        pi = 1,
        l1 = 0.01,
        )
solver_params = dict(
        tau_A = 1e9,
        tau_u = 1e1,
        tau_x = 1e4,
        T_u = 0,
        asynch=False,
        )

model = CTSCModel(**model_params)
model.A.data = loader.bases.t()
solver = CTSCSolver(model, loader, **solver_params)

solver.get_dir_path('bars')
#dir_path = get_timestamped_dir(load=True, base_dir='bars')
#solver = load_solver(dir_path, loader)

tspan = np.arange(1e4)

soln = solver.solve(tspan, out_N=1e2)
solver.save_soln(soln)
X = soln['x']
R = soln['r']
A = soln['A']

show_img_XRA(X, R, A, img_shape=(H, W))
