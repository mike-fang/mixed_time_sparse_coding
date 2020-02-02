import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis

# Define loader
H = W = 4
N_DIM = H * W
N_BATCH = 2 * (H * W)
N_DICT = H + W
PI = 0.3
bars_loader = BarsLoader(H, W, N_BATCH, p=PI, seed=42, numpy=True)

class BufferedLoader:
    def __init__(self, loader, buff_size=20, path=None):
        if path is None:
            path = './results/loader_buffer.npy'
        try:
            self.buffer = np.load('./results/loader_buffer.npy')
        except:
            self.buffer = np.array([loader() for _ in range(buff_size)])
            np.save('./results/loader_buffer.npy', self.buffer)
        self.loader=loader
        self.buff_size=buff_size
        self.buff_idx = 0
    def __call__(self, n_batch=None):
        X = th.tensor(self.buffer[self.buff_idx])
        self.buff_idx += 1
        if self.buff_idx % self.buff_size == 0:
            self.buff_idx = 0
        print(X.mean())
        return X

loader = BufferedLoader(bars_loader)
# DSC params
N_A = 10.5
N_S = 250
ETA_A = 0.02
ETA_S = 0.05




EXP = 'dsc'
#EXP = 'slsc'
EXP = 'lsc'
# model params
model_params = dict(
        n_dict=N_DICT,
        n_dim=N_DIM,
        n_batch=N_BATCH,
        positive=True,
        pi=PI if 'lsc' in EXP else 1,
        l1=.1,
        sigma=1.0,
        )

print(model_params)
LOAD = False
assert EXP in ['dsc', 'ctsc', 'asynch', 'lsc', 'slsc']
base_dir = f'bars_large_step_{EXP}'
# Define model, solver
model = CTSCModel(**model_params)
try:
    model.A.data = np.load('./A_untrained.npy')
    print('load')
except:
    np.save('./A_untrained.npy', model.A.data.numpy())
    print('save')
#model.A.data = bars_loader.bases.T

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
elif EXP == 'slsc':
    solver_params['spike_coupling'] = True
    solver_params['asynch'] = False
    solver_params['T_u'] = 1
    #model.pi = PI

# Load or make soln
solver = CTSCSolver(model, **solver_params)
#dir_path = solver.get_dir_path(base_dir)
solver.get_dir_path(base_dir)
soln = solver.solve(loader, soln_T=1)
solver.save_soln(soln)

assert False
A = soln['A'][:]
show_batch_img(A[-1].T, im_shape=(H, W))
plt.show()

X = soln['x'][:]
R = soln['r'][:]
A = soln['A'][:]

show_img_XRA(X, R, A, n_frames=1e2, img_shape=(H, W))
