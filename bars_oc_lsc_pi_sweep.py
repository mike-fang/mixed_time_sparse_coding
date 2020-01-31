import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img, plot_dict
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis

DICT = 'none'

# Define loader
H = W = 8
N_DIM = H * W
OC = 2
N_BATCH = 8 * (H * W)
N_DICT = OC * (H + W)
PI = 0.1
SIGMA = .5
LARGE = False
L1 = 1.0
loader = BarsLoader(H, W, N_BATCH, p=PI, sigma=SIGMA, l1=1)
NAME = 'no_norm_A'
if DICT == 'learned':
    NAME = 'learned_dict'
elif DICT == 'random':
    NAME = 'random_dict'

N_A = 400
N_S = 400
ETA_A = 0.05
ETA_S = 0.05

base_dir = f'bars_lsc_oc_{OC}'

# model params
model_params = dict(
        n_dict=N_DICT,
        n_dim=N_DIM,
        n_batch=N_BATCH,
        positive=True,
        pi=PI,
        l1=L1,
        sigma=SIGMA,
        )


# Define model, solver
for pi in np.arange(0.1, 0.4, 0.1):
    pi = round(float(pi), 2)
    name = f'pi_{pi:.3f}'.replace('.', 'p')
    print(name, pi, type(pi))

    model_params['pi'] = pi
    model = CTSCModel(**model_params)
    solver_params = CTSCSolver.get_dsc(model, n_A=N_A, n_s=N_S, eta_A=ETA_A, eta_s=ETA_S, return_params=True)
    solver_params['spike_coupling'] = False
    solver_params['asynch'] = False
    solver_params['T_u'] = 1
    try:
        model.A.data = np.load('./A_untrained.npy')
        print('load')
    except:
        np.save('./A_untrained.npy', model.A.data.numpy())
        print('save')

    solver = CTSCSolver(model, **solver_params)
    dir_path = solver.get_dir_path(base_dir, name=name, overwrite=True)
    soln = solver.solve(loader, soln_T=N_S, soln_offset=-1, normalize_A=False)
    solver.save_soln(soln)

    continue

    X = soln['x'][:]
    R = soln['r'][:]
    A = soln['A'][:]

    plot_dict(A[-1], (8, 8), int(OC * 2), 8)

    out_path = os.path.join(dir_path, 'evol.mp4')
    out_path = None
    show_img_XRA(X, R, A, out_file=out_path, n_frames=1e2, img_shape=(H, W))

