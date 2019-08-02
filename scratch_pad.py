import numpy as np
from glob import glob
import h5py

idx = slice(0, 100, 5)
print(idx.indices(4))
print(np.arange(100)[idx])
print(len(list(idx)))
