from mixed_t_sc import *
from discrete_sc import *
from helpers import *
import matplotlib.pylab as plt
import numpy as np
from matplotlib import animation
from visualization import show_img_evo

# Define energy
l1 = 2
l0 = .5
sigma = .2
energy_l0 = Energy_L0(sigma, l0, l1, positive=True)
energy_l1 = Energy_L1(sigma, l1, positive=True)

#Define loader
H = W = 4
n_batch = H + W
p = 0.05
loader = HVLinesLoader(H, W, n_batch, p=p)

# Hyper-params
n_sparse = H + W
n_dim = int(H * W)
mtsc_params = {
        'tau_s': 1e2,
        'tau_x': 3e3,
        'tau_A': 1e5,
        'mu_s': .1,
        'mu_A': .1,
        }
dsc_params = {
        'eta_s': 1e-2,
        'eta_A': 5e-2
        }

# Time range
T_RANGE = 1e6
T_STEPS = int(T_RANGE)
tspan = np.linspace(0, T_RANGE, T_STEPS, endpoint=False)


A0 = np.zeros((H + W, H, W))
for n in range(H):
    A0[n][n, :] = 1
for n in range(W):
    A0[n + H][:, n] = 1


A0 = A0.reshape((H + W, -1)).T
s0 = np.zeros((n_batch, n_sparse))
pA0 = np.zeros_like(A0)
ps0 = np.zeros_like(s0)

def train_mtsc(f_name=None, n_frames=1000, good_init=False, overwrite=False):
    try:
        if overwrite:
            raise Exception(f'Overwriting results in {f_name}')
        soln = load_model(f_name, soln_only=True)
    except:
        skip = int(max(len(tspan)//n_frames, 1))
        out_t = tspan[::skip]
        mtsc = MixedTimeSC(n_dim, n_sparse, energy_l0, im_shape=(H, W), **mtsc_params)
        if good_init:
            init = (s0, A0, ps0, pA0)
        else:
            init = None
        soln_dict = mtsc.train(loader, tspan, out_t=out_t, init_sA=init)
        soln = save_model(mtsc, soln_dict, loader, f_name, overwrite=True)
    return soln

def train_dsc(f_name=None, n_frames=1000, good_init=False, overwrite=False):
    try:
        if overwrite:
            raise Exception(f'Overwriting results in {f_name}')
        soln = load_model(f_name, soln_only=True)
    except:
        dsc = DiscreteSC(n_dim, n_sparse, energy_l1, im_shape=(H, W), **dsc_params)
        soln_dict = dsc.train(loader, n_iter=int(1e3), max_iter_s=int(1e2))
        #dsc = DiscreteSC(n_dim, n_sparse, eta_A, eta_s, n_batch, l0, l1, sigma, positive=False)
        #soln_dict = dsc.train(loader, n_iter=int(1e3), max_iter_s=int(1e2))
        soln = save_model(dsc, soln_dict, loader, f_name, overwrite=True)
    return soln

mtsc_dir = './results/hv_mtsc_4x4_p_005'
dsc_dir = './results/hv_dsc_4x4_l1_2'
soln = train_mtsc(mtsc_dir, n_frames=int(1e5))
#soln = train_dsc(dsc_dir, n_frames=int(1e5))
reshaped_params = soln.get_reshaped_params()
A = reshaped_params['A']
R = reshaped_params['R']
X = reshaped_params['X']

XRA = np.concatenate((X, R, A), axis=1)
show_img_evo(XRA, ratio = (H + W)/3, n_frames=100, out_file=None)
