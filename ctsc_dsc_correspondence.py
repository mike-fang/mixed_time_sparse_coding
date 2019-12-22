import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA
import matplotlib.pylab as plt

# Define loader
H = W = 4
N_DIM = H * W
N_BATCH = H * W
N_DICT = H + W
PI = 0.3
loader = BarsLoader(H, W, N_BATCH, p=PI, test=True)

# DSC params
N_A = 10
N_S = 10
ETA_A = 0.1
ETA_S = 0.1

# CTSC params
TAU_X = N_S
T_MAX = N_A * (N_S)
TAU_S = 1 / ETA_S
TAU_A = TAU_X / ETA_A

# Define model, solver
model_params = dict(
        n_dict=N_DICT,
        n_dim=N_DIM,
        n_batch=N_BATCH,
        positive=True,
        pi=1,
        l1 = 0.2,
        sigma = 1.0,
        )
solver_params = dict(
        tau_A = TAU_A,
        tau_u = TAU_S,
        tau_x = TAU_X,
        T_u = 0,
        asynch=False,
        spike_coupling=True,
        )

model = CTSCModel(**model_params)
solver = CTSCSolver(model, loader, **solver_params)
#model.A.data = loader.bases.t()

model.A.data *= 0
model.A.data[:N_DICT, :N_DICT] = th.eye(N_DICT)
model.u.data *= 0
model.u.data += 1

#solver.get_dir_path('bars')

def f(model):
    return
    print(model.energy(x))
tspan = np.arange(T_MAX)
f(model)
soln = solver.solve(T_MAX, out_N=1e2, callback_freq=1, callback_fn=f)

#solver.save_soln(soln)
X = soln['x']
R = soln['r']
A = soln['A']

#show_img_XRA(X, R, A, img_shape=(H, W))
A = model.A.data.numpy()
#print(A)
fig, axes = plt.subplots(nrows=2, ncols=4)
axes = [a for row in axes for a in row]
for n, ax in enumerate(axes):
    ax.imshow(A[:, n].reshape(H, W))
#plt.show()
