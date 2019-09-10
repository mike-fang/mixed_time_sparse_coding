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
        positive = False
        )
solver_params = [
        dict(params = ['s'], tau=1e2, mu=.1, T=1),
        dict(params = ['A'], tau=1e5, mu=.1),
        ]
init = dict(
        l1 = 1,
        rho = 2,
        nu = 0,
        )


tmax = 1e6

for _ in range(100):
    tau_s = 10 ** np.random.uniform(1, 3)
    tau_A = 10 ** np.random.uniform(4, 6)
    mu_s = max(np.random.uniform(-.2, 1), 0)
    mu_A = max(np.random.uniform(-.2, 1), 0)

    solver_params = [
            dict(params = ['s'], tau=tau_s, mu=mu_s, T=1),
            dict(params = ['A'], tau=tau_A, mu=mu_A),
            ]


    mtsc_solver = MTSCSolver(model_params, solver_params, dir_path='hv_lines', im_shape=(H, W))
    mtsc_solver.model.reset_params(init=init)
    mtsc_solver.set_loader(loader, tau_x=3e3)
    dir_path = mtsc_solver.dir_path
    soln = mtsc_solver.start_new_soln(tmax=tmax, n_soln=10000, t_save=1e5)

    plt.plot(soln['energy'])
    plt.savefig(os.path.join(dir_path, 'energy.pdf'))

    reshaped_params = soln.get_reshaped_params()
    A = reshaped_params['A']
    R = reshaped_params['r']
    X = reshaped_params['x']
    XRA = np.concatenate((X, R, A), axis=1)

    show_img_evo(XRA, ratio = (H + W)/3, n_frames=100, out_file=os.path.join(dir_path, 'evolution.mp4'))