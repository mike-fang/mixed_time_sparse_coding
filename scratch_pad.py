import numpy as np
bins = np.arange(100.)
bins = np.insert(bins, 1, 1e-5)
print(bins)
