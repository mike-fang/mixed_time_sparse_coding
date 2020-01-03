from ctsc import get_timestamped_dir
from visualization import show_img_XRA
import h5py
import os.path
import numpy as np
import matplotlib.pylab as plt

DIM = 16

base_dir = 'vh_1T_dim_16'
dir_path = get_timestamped_dir(load=True, base_dir=base_dir)
print(dir_path)
soln = h5py.File(os.path.join(dir_path, 'soln.h5'))

for n in soln:
    print(n)
X = soln['x'][:]
N, _, _ = X.shape
print(X.shape)

X = soln['x'][:]
R = soln['r'][:]
A = soln['A'][:]

out_path = os.path.join(dir_path, 'evol.mp4')
show_img_XRA(X, R, A, n_frames=1e2, img_shape=(DIM, DIM), out_file=out_path, N_batch=20, N_dict=20)
