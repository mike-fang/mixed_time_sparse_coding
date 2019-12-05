import h5py
import torch as th
import numpy as np


h5_file = h5py.File('./vanhateren_imc/images.h5', 'a')
img_ds = h5_file['100']
stdev = np.std(img_ds[:])
mean = np.mean(img_ds[:])
print(img_ds[:].min())
