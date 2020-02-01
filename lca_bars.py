import numpy as np
from loaders import BarsLoader
from tqdm import tqdm
import matplotlib.pylab as plt
from ctsc import CTSCSolver
from collections import defaultdict
from visualization import show_img_XRA, show_batch_img
import h5py
import os.path
import yaml
from lca import *


H = W = 8
N_DIM = H * W
N_BATCH = 10 * (H * W)
N_DICT = H + W
PI = 0.3
SIGMA = .5
loader = BarsLoader(H, W, N_BATCH, sigma=0.5, p=PI, numpy=True)

N_A = 400
N_S = 400
eta_A = 0.05
eta_S = 0.05

U0 = 1

lca = LCAModel(n_dim=N_DIM, n_dict=N_DICT, n_batch=N_BATCH, u0=U0, sigma=SIGMA, positive=True)
solver = LCASolver(lca, N_A, N_S, eta_A, eta_S)
dir_path = solver.get_dir_path('bars_lca', name='no_norm_A', overwrite=True)
soln = solver.solve(loader, soln_T=N_S, soln_offset=-1, normalize_A=False)
solver.save_soln(soln)
X = soln['x'][:]
R = soln['r'][:]
A = soln['A'][:]

#out_path = os.path.join(dir_path, 'evol.mp4')
if False:
    out_path = None
    show_img_XRA(X, R, A, out_file=out_path, n_frames=1e2, img_shape=(H, W))

A = soln['A'][-1]
fig, axes = plt.subplots(nrows=4, ncols=4)
axes = [a for row in axes for a in row]
for n, ax in enumerate(axes):
    ax.imshow(A[:, n].reshape(H, W))
plt.show()
