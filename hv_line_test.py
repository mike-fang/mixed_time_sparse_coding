from mixed_t_sc import MixedTimeSC, save_soln
from discrete_sc import DiscreteSC
from loaders import *
import matplotlib.pylab as plt
import numpy as np
from matplotlib import animation
from visualization import show_img_evo

# Data params
H = W = 5
p = .1
n_batch = 10
loader = HVLinesLoader(H, W, n_batch, p=p)

def sA_HV_bases():
    s = np.zeros((n_batch, n_sparse))
    A = np.zeros((n_dim, n_sparse))
    A[:, :H + W] = loader.bases.T
    return s, A

# Model Params
n_dim = int(H * W)
n_sparse = 10
l1 = .5
l0 = .5
sigma = .2

# MTSC Params
tau_s = 1e2
tau_x = 5e3
tau_A = 1e5
T_RANGE = 1e3
T_STEPS = int(T_RANGE)
tspan = np.linspace(0, T_RANGE, T_STEPS, endpoint=False)

# DSC Params
eta_A = 1e-2
eta_s = 1e-3

def train_mtsc(f_name=None):
    mtsc = MixedTimeSC(n_dim, n_sparse, tau_s, tau_x, tau_A, l0, l1, sigma, n_batch, positive=True)
    soln_dict = mtsc.train(loader, tspan, init_sA=None, no_noise=False)
    soln = Solutions_H5(f_name, soln_dict, im_shape=(H, W), overwrite=True)
    return soln


def train_dsc():
    l0 = 0.0
    dsc = DiscreteSC(n_dim, n_sparse, eta_A, eta_s, n_batch, l0, l1, sigma, positive=False)
    soln_dict = dsc.train(loader, n_iter=int(1e3), max_iter_s=int(1e2))
    soln = Solutions(soln_dict, im_shape=(H, W))
    soln.save(f_name='./results/hv_line_dsc.soln')

soln = train_mtsc('./results/hv_mtsc.soln')
reshaped_params = soln.get_reshaped_params()
A = reshaped_params['A']
R = reshaped_params['R']
X = reshaped_params['X']

XRA = np.concatenate((X, R, A), axis=1)
show_img_evo(XRA, n_frames=100, ratio = 10/3)

