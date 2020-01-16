import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis

def plot_dict():
    A = loader.bases.reshape(-1, 8, 8)
    fig, axes = plt.subplots(2, 8, figsize=(8, 2))
    axes = [ax for row in axes for ax in row]
    for n, ax in enumerate(axes):
        ax.imshow(A[n], cmap='Greys_r')
        ax.set_xlabel(rf'$A_{{{n}}}$')
        ax.set_xticks([])
        ax.set_yticks([])
    #fig.suptitle('Bars Dictionary')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

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

# Define loader
H = W = 8
N_DIM = H * W
N_BATCH = 2 * (H * W)
N_DICT = H + W
PI = 0.3
SIGMA = .5
LARGE = False
N_S = 200
L1 = 1
loader = BarsLoader(H, W, N_BATCH, p=PI, sigma=SIGMA, l1=L1)
EXP = 'dsc'

N_A = 1000
#N_A = 100
N_S = N_S
ETA_A = 0.03
#ETA_A = 1e-9
ETA_S = 0.02


# model params
model_params = dict(
        n_dict=N_DICT,
        n_dim=N_DIM,
        n_batch=N_BATCH,
        positive=True,
        pi=1.,
        l1=L1,
        sigma=SIGMA,
        )

for EXP in ['dsc', 'ctsc', 'asynch', 'lsc']:
    print(EXP)
    base_dir = f'bars_{EXP}'

    # Define model, solver
    model = CTSCModel(**model_params)
    try:
        model.A.data = np.load('./A_untrained.npy')
        print('load')
    except:
        np.save('./A_untrained.npy', model.A.data.numpy())
        print('save')
    #model.A.data = loader.bases.T
    #model.A.data = th.tensor(np.load('./A0.npy'))
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
        solver_params['asynch'] = True
        solver_params['T_u'] = 1
        model.pi = PI

    # Load or make soln
    solver = CTSCSolver(model, **solver_params)
    dir_path = solver.get_dir_path(base_dir)
    #solver.get_dir_path(base_dir)
    soln = solver.solve(loader, soln_T=N_S, soln_offset=-1, out_mse=True)
    solver.save_soln(soln)

    t = soln['mse_t']
    mse = soln['mse']

    X = soln['x'][:]
    R = soln['r'][:]
    A = soln['A'][:]

    out_path = os.path.join(dir_path, 'evol.mp4')
    show_img_XRA(X, R, A, out_file=out_path, n_frames=1e2, img_shape=(H, W))
    assert False
    plt.show()
