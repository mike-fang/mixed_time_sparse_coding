import numpy as np
import torch as th
from time import time

H = W = 4
n_batch = 1
S = th.Tensor(n_batch, W + H).bernoulli_()
S *= th.Tensor(n_batch, W + H).uniform_()
print(S)

