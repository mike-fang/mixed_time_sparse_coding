import numpy as np

N = 3
A = np.arange(6*3).reshape((N, 2, 3))
print(A)
print(A.reshape((-1, 2)))
