from mt_sc_direct_implementation import MixT_SC, save_soln
from loaders import HVLinesLoader
import matplotlib.pylab as plt
import numpy as np
from matplotlib import animation

H = W = 5
p = .2
n_batch = 10

tau_s = 1e2
tau_x = 1e3
tau_A = 1e4
T_RANGE = 1e5
T_STEPS = int(T_RANGE)
tspan = np.linspace(0, T_RANGE, T_STEPS, endpoint=False)

n_dim = int(H * W)
n_sparse = n_dim * 2
n_sparse = 10

l1 = .5
l0 = .9
sigma = .2

loader = HVLinesLoader(H, W, n_batch, p=p)

mtsc = MixT_SC(n_dim, n_sparse, tau_s, tau_x, tau_A, l0, l1, sigma, n_batch, positive=False)

def sA_HV_bases():
    s = np.zeros((n_batch, n_sparse))
    A = np.zeros((n_dim, n_sparse))
    A[:, :H + W] = loader.bases.T
    return s, A

s, A = sA_HV_bases()
solns = mtsc.solve(loader, tspan, init_sA=None, no_noise=False)
save_soln(solns, im_shape=(H, W), f_name='./results/hv_line_HV.h5py', overwrite=True)
#save_soln(solns, im_shape=(H, W), f_name='./results/hv_line_HV_init.h5py', overwrite=True)
