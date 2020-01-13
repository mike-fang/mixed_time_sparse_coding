import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis

N_BATCH = 5
class DummyLoader():
    def __init__(self):
        pass
    def __call__(self, n_batch=N_BATCH):
        return th.zeros(n_batch, 1)

# model params
model_params = dict(
        n_dict=1,
        n_dim=1,
        n_batch=N_BATCH,
        positive=True,
        pi= .5,
        l1=.3,
        sigma=1e9,
        )

solver_params = dict(
        tau_A = 1e9,
        tau_u = 50,
        tau_x = 1e9,
        asynch = False,
        t_max = 10000,
        )

l1_list = [.1, .5, 1.]

s_solns = []
for l1 in l1_list:
    model_params['l1'] = l1
    loader = DummyLoader()
    model = CTSCModel(**model_params)
    solver = CTSCSolver(model, **solver_params)
    soln = solver.solve(loader, soln_T=1)
    s = soln['s'][:]
    plt.plot(np.squeeze(s))
    plt.show()
s_solns = np.arange()

assert False
nz_data = []
l1_list = [.2, .5, 1]
colors = ['r', 'g', 'b']
for l1 in l1_list:
    model.l1 = l1
    solver = CTSCSolver(model, **solver_params)
    soln = solver.solve(loader, soln_T=1)

    s = soln['s'][:]
    non_zero = (s > 0)
    T, _, _ = s.shape
    nz_list = []
    for t in np.linspace(0, T, 100, dtype=int):
        nz_list.append(non_zero[:t].mean())

    nz_data.append(nz_list)

nz_data = np.array(nz_data)
for n , nz in enumerate(nz_data):
    plt.plot(nz, color=colors[n], label=l1_list[n])
plt.show()
