import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img, plot_dict
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis


# Define loader
H = W = 8
N_DIM = H * W
OC = 2
N_BATCH = 10 * (H * W)
N_DICT = OC * (H + W)
PI = 0.3
SIGMA = .5
LARGE = False
L1 = 1.0
loader = BarsLoader(H, W, N_BATCH, p=PI, sigma=SIGMA, l1=L1)
NAME = 'no_norm_A'
NAME = 'learn_pi'
NAME = 'fixed_pi'


N_A = 400
N_S = 400
ETA_A = 0.1
ETA_S = 0.05

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

base_dir = f'bars_learn_pi'

# Define model, solver
model = CTSCModel(**model_params)

try:
    model.A.data = np.load('./A_untrained.npy')
    print('load')
except:
    np.save('./A_untrained.npy', model.A.data.numpy())
    print('save')
solver_params = CTSCSolver.get_dsc(model, n_A=N_A, n_s=N_S, eta_A=ETA_A, eta_s=ETA_S, return_params=True)
solver_params['spike_coupling'] = False
solver_params['asynch'] = True
solver_params['T_u'] = 1
solver_params['tau_u0'] = 1e30
#model.pi = PI
#model.A.data = loader.bases.T/2

# Load or make soln
solver = CTSCSolver(model, **solver_params)
dir_path = solver.get_dir_path(base_dir, name=NAME, overwrite=True)
#solver.get_dir_path(base_dir)
soln = solver.solve(loader, soln_T=N_S, soln_offset=-1, out_energy=False, normalize_A=False, norm_A_init=.1)
solver.save_soln(soln)

X = soln['x'][:]
R = soln['r'][:]
A = soln['A'][:]

plot_dict(A[-1], (8, 8), int(OC * 2), 8)
plt.show()

out_path = os.path.join(dir_path, 'evol.mp4')
out_path = None
#show_img_XRA(X, R, A, out_file=out_path, n_frames=1e2, img_shape=(H, W))
