import torch as th
import numpy as np
from lca import *
from loaders import VanHaterenSampler
from visualization import show_img_XRA
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis

DIM = 8
OC = 2
BATCH_FRAC = 0.5

H = W = DIM
N_DIM = H * W
N_BATCH = int(N_DIM * BATCH_FRAC)
N_DICT = int(OC * N_DIM)
base_dir = f'vh_dim_{DIM}_lca'

loader = VanHaterenSampler(H, W, N_BATCH)
model_params = dict(
        n_dict=N_DICT,
        n_dim=N_DIM,
        n_batch=N_BATCH,
        positive=True,
        u0 = .2,
        )
solver_params = dict(
    n_A = 4000,
    n_s = 50,
    eta_A = 0.02,
    eta_s = 0.01,
)
model = LCAModel(**model_params)
solver = LCASolver(model, **solver_params)
dir_path = solver.get_dir_path(base_dir)
soln = solver.solve(loader, soln_T=solver_params['n_s'] * 10, soln_offset=-1)
solver.save_soln(soln)
