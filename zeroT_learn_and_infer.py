from ctsc import *
import numpy as np
import torch as th
from loaders import BarsLoader, VanHaterenSampler
from visualization import show_img_XRA
import os.path

# Define loader
H = W = 4
N_BATCH = H + W
LOADER_PI = 0.2
MODEL_PI = 1
loader = BarsLoader(H, W, N_BATCH, p=LOADER_PI)

model_params = dict(
        n_dict = H * W,
        n_dim = H * W,
        n_batch = N_BATCH,
        pi=MODEL_PI,
        positive=True,
        l1 = 1,
        )
solver_params = dict(
        tau_A = 5e2 * N_BATCH,
        tau_u = 5e1,
        tau_x = 5e2,
        T_u = 0.,
        asynch=False,
        )

model = CTSCModel(**model_params)
solver = CTSCSolver(model, loader, **solver_params)
solver.get_dir_path('bars')

# Run solver
tspan = np.arange(1e5)
soln = solver.solve(tspan, out_N=1e2)

# Save and visualize soln
solver.save_soln(soln)
X = soln['x']
R = soln['r']
A = soln['A']

out_path = os.path.join(solver.dir_path, 'evol.mp4')
show_img_XRA(X, R, A, img_shape=(H, W))
