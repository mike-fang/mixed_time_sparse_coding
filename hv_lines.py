from loaders import HVLinesLoader
import matplotlib.pylab as plt
import numpy as np
import torch as th
from visualization import show_img_evo
from mtsc import *

#Define loader
H = W = 4
n_batch = H + W
p = 0.05
loader = HVLinesLoader(H, W, n_batch, p=p)

model_params = dict(
        n_dict = H + W,
        n_dim = H * W,
        n_batch = n_batch,
        positive = True
        )
solver_params = [
        dict(params = ['s'], tau=1e2, T=1),
        dict(params = ['A'], tau=1e5),
        ]
init = dict(
        l1 = 2,
        rho = 5,
        nu = 0,
        )

tmax = 1e4
print(loader().shape)

if False:
    mtsc_solver = MTSCSolver(model_params, solver_params, dir_path='hv_lines', im_shape=(H, W))
    mtsc_solver.model.reset_params(init=init)
    mtsc_solver.set_loader(loader, tau_x=3e3)
    soln = mtsc_solver.start_new_soln(tmax=tmax, n_soln=10000)
else:
    mtsc_solver = MTSCSolver.load('hv_lines')
    soln = mtsc_solver.load_soln()

    reshaped_params = soln.get_reshaped_params()
    A = reshaped_params['A']
    R = reshaped_params['r']
    X = reshaped_params['x']

    XRA = np.concatenate((X, R, A), axis=1)
    show_img_evo(XRA, ratio = (H + W)/3, n_frames=100, out_file=None)
