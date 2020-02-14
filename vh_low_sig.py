import torch as th
import numpy as np
from ctsc import *
from loaders import VanHaterenSampler
from visualization import show_img_XRA, plot_dict
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis

def load_previous(base_dir):
    dir_path = get_timestamped_dir(load=True, base_dir=base_dir)
    analysis = SolnAnalysis(dir_path)
    A = analysis.A[-1]
    pi = analysis.pi[-1]
    return th.tensor(A).to('cuda'), pi

DIM = 8
OC = 6
EXP = 'learn_pi'
CONTINUE = True
BATCH_FRAC = 2
H = W = DIM
N_DIM = H * W
N_BATCH = int(N_DIM * BATCH_FRAC)
N_DICT = int(OC * N_DIM)
FIX_PI = False

N_S = 400
# DSC params
dsc_params = dict(
    n_A = 3000,
    n_s = N_S,
    eta_A = 0.01,
    eta_s = 0.05,
)

model_params = dict(
        n_dict=N_DICT,
        n_dim=N_DIM,
        n_batch=N_BATCH,
        positive=True,
        pi=.30,
        l1=1,
        sigma=.3,
        )

base_dir = f'vh_{EXP}_dim_{DIM}_oc_{OC}'.replace('.', 'p')
if FIX_PI:
    base_dir += '_fixed_pi'
loader = VanHaterenSampler(H, W, N_BATCH)

solver_params = dsc_solver_param(**dsc_params)
solver_params['spike_coupling'] = False
solver_params['asynch'] = False
solver_params['T_u'] = 1
if not FIX_PI:
    solver_params['tau_u0'] = 2e8

# Define model, solver
if CONTINUE:
    A, pi = load_previous(base_dir)
    model_params['pi'] = pi
model = CTSCModel(**model_params)
model = model.to('cuda')
model.A.data *= .3
if CONTINUE:
    model.A.data = A

solver = CTSCSolver(model, **solver_params)
dir_path = solver.get_dir_path(base_dir)
soln = solver.solve(loader, soln_T=N_S * 20, soln_offset=-1, normalize_A=False, report_norm_pi=True)
solver.save_soln(soln)

A = soln['A'][:]
plot_dict(A[-1], (8, 8), int(OC * 2), 8)
plt.show()

