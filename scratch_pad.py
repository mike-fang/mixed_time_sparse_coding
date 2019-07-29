import numpy as np
import h5py

a = np.arange(2*3*4).reshape((4, 3, 2))
b = np.arange(3*4).reshape((3, 4))
print(b@a[:, None])

