import numpy as np
from numpy.linalg import svd

A = np.array([
        [1, 1],
        [0, 0]
        ]
        )

_, S, _ = (svd(A))
print(S)
