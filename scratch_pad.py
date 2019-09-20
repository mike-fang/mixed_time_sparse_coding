import torch as th
import numpy as np
from time import time

N = 10000

X = th.FloatTensor(N, N).normal_(0, 30.5)
print(X.shape)
