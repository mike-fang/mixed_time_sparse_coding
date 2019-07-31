import numpy as np

X = np.random.randn(5, 4, 3)
print(np.transpose(X, None).shape)
