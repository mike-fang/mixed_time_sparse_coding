import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis

exp = 'lsc'
base_dir = f'bars_{exp}'
dir_path = get_timestamped_dir(load=True, base_dir=base_dir)
analysis = SolnAnalysis(dir_path)

A_soln = analysis.soln['A']
#print( (10000 < analysis.time).argmax()) 
def idx_from_t(t, soln):
    time = soln['t']
    if t > time.max():
        print('Warning: time given greater than max t_range')
    return (t < time).argmax()
def A_at_t(t, soln):
    idx = idx_from_t(t, soln)
    A = soln['A'][idx]
    return A

H = W = 8
N_BATCH = 2 * (H * W)
PI = 0.3
SIGMA = .5
N_S = 200
L1 = 1

solver = CTSCSolver.load(dir_path=dir_path)
loader = BarsLoader(H, W, N_BATCH, p=PI, sigma=SIGMA, l1=L1)

model_sigma = solver.model.sigma
solver.model.sigma = 1e20
solver.asynch = False
N = solver.model.n_batch
D = solver.model.n_dim

mean_nnl = []
for A in tqdm(A_soln[::10]):
    solver.model.A.data = th.tensor(A)
    soln = solver.solve(loader, tmax=N_S, soln_T=1)
    R = soln['r'][:]
    X = soln['x'][0]

    _, _, D = R.shape
    R = R[::10].reshape((-1, D))
    diff = X[None, :, :] - R[:, None, :]
    
    mse = np.mean(diff**2, axis=(2))
    root_px = np.exp(-0.5 * mse / model_sigma**2) / (model_sigma * (2 * np.pi)**0.5)
    mean_px = (root_px**D).mean(axis=0) + 1E-50
    nlog_px = - np.log(mean_px)
    mean_nnl.append(np.nanmean(nlog_px))

print(mean_nnl)
plt.plot(mean_nnl)
plt.show()
