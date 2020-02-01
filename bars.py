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


DICT = 'none'
EXP = 'dsc'

# Define loader
H = W = 8
N_DIM = H * W
OC = 1
N_BATCH = 10 * (H * W)
N_DICT = OC * (H + W)
PI = 0.3
SIGMA = .5
LARGE = False
N_S = 1000
L1 = 1.0
loader = BarsLoader(H, W, N_BATCH, p=PI, sigma=SIGMA, l1=L1)
NAME = 'no_norm_A'
if DICT == 'learned':
    NAME = 'learned_dict'
elif DICT == 'random':
    NAME = 'random_dict'

N_A = 400
if DICT in ['learned', 'random']:
    N_A = 100
N_S = N_S
ETA_A = 0.05
if DICT in ['learned', 'random']:
    ETA_A = 1e-20
ETA_S = 0.05


# model params
model_params = dict(
        n_dict=N_DICT,
        n_dim=N_DIM,
        n_batch=N_BATCH,
        positive=True,
        pi=1,
        l1=L1,
        sigma=SIGMA,
        )

base_dir = f'bars_{EXP}'

# Define model, solver
model = CTSCModel(**model_params)
try:
    model.A.data = np.load('./A_untrained.npy')
    print('load')
except:
    np.save('./A_untrained.npy', model.A.data.numpy())
    print('save')
if DICT == 'learned':
    model.A.data = loader.bases.T
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
    model.pi = PI

# Load or make soln
solver = CTSCSolver(model, **solver_params)
dir_path = solver.get_dir_path(base_dir, name=NAME, overwrite=True)
#solver.get_dir_path(base_dir)
soln = solver.solve(loader, soln_T=N_S, soln_offset=-1, out_mse=True, out_energy=False, normalize_A=False)
solver.save_soln(soln)

t = soln['mse_t']
mse = soln['mse']

X = soln['x'][:]
R = soln['r'][:]
A = soln['A'][:]

plot_dict(A[-1], (8, 8), int(OC * 2), 8)

out_path = os.path.join(dir_path, 'evol.mp4')
out_path = None
#show_img_XRA(X, R, A, out_file=out_path, n_frames=1e2, img_shape=(H, W))
