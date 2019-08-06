from mixed_t_sc import *
from discrete_sc import DiscreteSC
from loaders import *
import matplotlib.pylab as plt
import numpy as np
from matplotlib import animation
from visualization import show_img_evo

# Define energy
l1 = .3
l0 = .3
sigma = .5
energy = Energy_L0(sigma, l0, l1, positive=True)

#Define loader
H = W = 5
n_batch = 10
p = 0.1
loader = HVLinesLoader(H, W, n_batch, p=p)

# Hyper-params
n_sparse = 20
n_dim = int(H * W)
params = {
        'tau_s': 1e2,
        'tau_x': 3e3,
        'tau_A': 1e5,
        'mu_s': .0,
        'mu_A': .05,
        }

# Time range
T_RANGE = 1e6
T_STEPS = int(T_RANGE)
tspan = np.linspace(0, T_RANGE, T_STEPS, endpoint=False)


def train_mtsc(f_name=None, n_frames=1000):
    skip = int(max(len(tspan)//n_frames, 1))
    out_t = tspan[::skip]
    mtsc = MixedTimeSC(n_dim, n_sparse, energy, **params)
    soln_dict = mtsc.train(loader, tspan, out_t=out_t, init_sA=None)
    soln = save_model(mtsc, soln_dict, loader, f_name, overwrite=True)
    return soln

def train_dsc():
    # DSC Params
    eta_A = 1e-2
    eta_s = 1e-3
    l0 = 0.0
    dsc = DiscreteSC(n_dim, n_sparse, eta_A, eta_s, n_batch, l0, l1, sigma, positive=False)
    soln_dict = dsc.train(loader, n_iter=int(1e3), max_iter_s=int(1e2))
    soln = Solutions(soln_dict, im_shape=(H, W))
    soln.save(f_name='./results/hv_line_dsc.soln')

soln = train_mtsc('./results/hv_mtsc_momentum', n_frames=10000)
soln = Solutions_H5.load_h5('./results/hv_mtsc_momentum/soln.h5')
reshaped_params = soln.get_reshaped_params()
A = reshaped_params['A']
R = reshaped_params['R']
X = reshaped_params['X']

XRA = np.concatenate((X, R, A), axis=1)
show_img_evo(XRA, ratio = 10/4, n_frames=100)
