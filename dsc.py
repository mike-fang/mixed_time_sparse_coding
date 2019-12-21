import torch as th
import numpy as np
from loaders import BarsLoader
from torch.nn import Parameter

H = W = 4
N_DIM = H * W
N_BATCH = H * W
N_DICT = H + W
PI = 0.3
loader = BarsLoader(H, W, N_BATCH, p=PI)

N_A = 2500
N_S = 20
eta_A = 0.1
eta_S = 0.1

sigma = 1.0
l1 = 0.2

A = Parameter(th.Tensor(N_DIM, N_DICT))
s = Parameter(th.Tensor(N_DICT, N_BATCH))
self.A.data.normal_()
self.u.data.normal_()

def energy(A, s, x):
    recon = 0.5 * (((A@s).t() - x)**2).sum()
    sparse = l1 * th.abs(s).sum()
    return recon/sigma**2 + l1 * sparse
