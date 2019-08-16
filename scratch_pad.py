import torch as th
from torch.nn import Parameter
from time import time


N = int(1e7)
dW = th.zeros(N)
t0 = time()
dW.normal_()
print(time() - t0)

t0 = time()
dW.normal_()
print(time() - t0)
