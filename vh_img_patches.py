from loaders import VanHaterenSampler
import matplotlib.pylab as plt
import numpy as np
import torch as th
from visualization import show_img_evo, show_img_XRA
from mtsc import *

#Define loader
H = W = 8
n_batch = 64
p = 0.05
loader = VanHaterenSampler(H, W, n_batch)
TAU_S = int(2e2)
TAU_X = int(5e2)
TAU_A = int(1e3)
TAU_RATIO = 10
ALPHA = 0.2
OC = 1
n_dict = int(H*W*OC)
tmax = 1e4

def coupling_A(t_):
    return (1 + (TAU_RATIO - 1) * np.exp(-t_/ALPHA))**(-1)
coupling_A = 'const'

model_params = dict(
        n_dict = n_dict,
        n_dim = H * W,
        n_batch = n_batch,
        positive = True
        )
solver_params = [
        dict(params = ['s_data'], tau=TAU_S, T=0),
        dict(params = ['A'], tau=TAU_A*n_batch),
        ]
init = dict(
        l1 = 1,
        pi = 1,
        sigma = 1,
        )

mtsc_solver = MTSCSolver(model_params, solver_params, base_dir='vh_patches', im_shape=(H, W), coupling=coupling_A)
mtsc_solver.model.reset_params(init=init)
mtsc_solver.set_loader(loader, tau_x=TAU_X)
soln = mtsc_solver.start_new_soln(tmax=tmax, n_soln=10000, t_save=1e5, rand_tau_x=True)
#soln = mtsc_solver.load('dsc_4x4')

dir_path = mtsc_solver.dir_path
#plt.plot(soln['energy'])
#plt.savefig(os.path.join(dir_path, 'energy.pdf'))

reshaped_params = soln.get_reshaped_params()
X = reshaped_params['x_data']
R = reshaped_params['r_data']
A = reshaped_params['A']

out_path = os.path.join(dir_path, 'evol.mp4')
#out_path = None
show_img_XRA(X, R, A, n_frames=100, out_file=out_path)
