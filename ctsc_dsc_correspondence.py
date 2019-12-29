import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis

# Define loader
H = W = 4
N_DIM = H * W
N_BATCH = H * W
N_DICT = H + W
PI = 0.3
loader = BarsLoader(H, W, N_BATCH, p=PI)

# DSC params
N_A = 100
N_S = 25
ETA_A = 0.5
ETA_S = 0.1

# Define model, solver
model_params = dict(
        n_dict=N_DICT,
        n_dim=N_DIM,
        n_batch=N_BATCH,
        positive=True,
        pi=1.,
        l1 = 0.2,
        sigma = 1.0,
        )

try:
    dir_path = get_timestamped_dir(load=True, base_dir='bars_dsc')
    analysis = SolnAnalysis(dir_path)
    #print(analysis.solver.tau_x)
    time = analysis.soln['t'][:]
    where_load = (time % analysis.solver.tau_x == 0)
    energy, recon = analysis.energy()
    plt.plot(analysis.soln['t'][where_load], energy[where_load])
    plt.plot(analysis.soln['t'][where_load], recon[where_load])
    plt.show()
except:
    model = CTSCModel(**model_params)
    solver = CTSCSolver.get_dsc(model, n_A=N_A, n_s=N_S, eta_A=ETA_A, eta_s=ETA_S)
    solver.spike_coupling=False
    solver.asynch=True
    solver.get_dir_path('bars_asynch')
    soln = solver.solve(loader, out_N=1e3, save_N=1)
    solver.save_soln(soln)

    X = soln['x']
    R = soln['r']
    A = soln['A']

    show_img_XRA(X, R, A, img_shape=(H, W))
