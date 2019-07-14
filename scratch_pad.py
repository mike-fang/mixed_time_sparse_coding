import numpy as np

A = np.arange(12).reshape((3, 4))
print(A / np.linalg.norm(A, axis=0))
