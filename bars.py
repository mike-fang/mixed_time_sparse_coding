import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img, plot_dict
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis

def plot_samples(pi=.3, l1=1, sigma=0):
    loader = BarsLoader(H, W, 16, l1=l1, p=pi, sigma=sigma, numpy=True)
    x = loader().reshape((-1, 8, 8))
    fig, axes = plt.subplots(2, 8, figsize=(8, 2))
    axes = [ax for row in axes for ax in row]
    for n, ax in enumerate(axes):
        ax.imshow(x[n], cmap='Greys_r')
        #ax.set_xlabel(rf'$A_{{{n}}}$')
        ax.set_xticks([])
        ax.set_yticks([])
    #fig.suptitle(fr'Bars Samples: $\lambda = {l1}, \pi = {PI}, \sigma = {sigma}$')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'
DICT = None
EXP = 'lsc'

# Define loader
H = W = 8
N_DIM = H * W
OC = 1
N_BATCH = 8 * (H * W)
N_DICT = OC * (H + W)
PI = .3
SIGMA = .5
LARGE = False
L1 = 1.0
loader = BarsLoader(H, W, N_BATCH, p=PI, sigma=SIGMA, l1=L1)
NORM_A = False
NAME = 'no_norm_A'
NAME = None
if DICT == 'learned':
    NAME = 'learned_dict'
elif DICT == 'random':
    NAME = 'random_dict'

N_S = 200
N_A = 800
if DICT in ['learned', 'random']:
    N_A = 100
N_S = N_S
ETA_A = 0.1
if DICT in ['learned', 'random']:
    ETA_A = 1e-99
ETA_S = 0.05


# model params
model_params = dict(
        n_dict=N_DICT,
        n_dim=N_DIM,
        n_batch=N_BATCH,
        positive=True,
        pi=PI if EXP == 'lsc' else 1,
        l1=L1,
        sigma=SIGMA,
        )

base_dir = f'bars_{EXP}'

# Define model, solver
model = CTSCModel(**model_params)
model = model.to(DEVICE)

model.A.data *= th.linspace(0.3, 1.2, N_DICT)[None, :]
#model.A.data *= 1.2
if DICT == 'learned':
    model.A.data = loader.bases.t().to(DEVICE)
solver_params = CTSCSolver.get_dsc(model, n_A=N_A, n_s=N_S, eta_A=ETA_A, eta_s=ETA_S, return_params=True)
if EXP == 'dsc':
    pass
elif EXP == 'ctsc':
    solver_params['spike_coupling'] = False
elif EXP == 'asynch':
    solver_params['spike_coupling'] = False
    solver_params['asynch'] = True
elif EXP == 'lsc':
    solver_params['spike_coupling'] = False
    solver_params['asynch'] = False
    solver_params['T_u'] = 1

# Load or make soln
solver = CTSCSolver(model, **solver_params)
dir_path = solver.get_dir_path(base_dir, name=NAME, overwrite=True)
#solver.get_dir_path(base_dir)
soln = solver.solve(loader, soln_T=N_S, soln_offset=-1, out_energy=False, normalize_A=NORM_A, report_norm_pi=True)
solver.save_soln(soln)

X = soln['x'][:]
R = soln['r'][:]
A = soln['A'][:]

plot_dict(A[-1], (8, 8), int(OC * 2), 8)

out_path = os.path.join(dir_path, 'evol.mp4')
out_path = None
#show_img_XRA(X, R, A, out_file=out_path, n_frames=1e2, img_shape=(H, W))
