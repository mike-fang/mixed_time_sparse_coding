from loaders import BarsLoader
import matplotlib.pylab as plt
import numpy as np
import torch as th
from visualization import show_img_evo, show_img_XRA
from ctsc import *
from L0_sparsity import *

H = W = 4
N_BATCH = H + W
PI = 0.2
loader = BarsLoader(H, W, N_BATCH, p=PI)

model_params = dict(
        n_dict = H * W,
        n_dim = H * W,
        n_batch = N_BATCH,
        positive=True,
        )
solver_params = dict(
        tau_A = 5e3,
        tau_u = 1e2,
        tau_x = 1e3,
        asynch=True,
        )

model = CTSCModel(**model_params)
solver = CTSCSolver(model, loader, **solver_params)

dir_path = get_timestamped_dir(load=True, base_dir='bars')
solver = load_solver(dir_path, loader)

tspan = np.arange(1e3)

soln = solver.solve(tspan, out_N=1e2)
solver.save_soln(soln, 'bars')
X = soln['x']
R = soln['r']
A = soln['A']

show_img_XRA(X, R, A, img_shape=(H, W))

