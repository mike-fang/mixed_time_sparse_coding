import torch as th
from torch.nn.parameter import Parameter
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt
from matplotlib import animation
import os.path as osp
from time import time
import random

def get_param(size, tau, T=1, init=None):
    param = Parameter(th.zeros(size))
    if init is not None:
        if not isinstance(init, th.Tensor):
            try:
                init = th.tensor(init)
            except:
                init = th.tensor([init])
        param.data = init
    if tau is None:
        freq = 0
        param.requires_grad = False
    else:
        freq = 1/tau
    param.freq = freq
    param.T = T
    return param
def L2_reconstr_loss(A, s, x, sigma):
    return 0.5 * th.norm(x - A @ s)**2 / sigma**2
def E(params):
    A = params['A']
    s = params['s']
    x = params['x']
    sigma = params['sigma']
    l0 = params['l0']
    l1 = params['l1']
    s0 = -th.log(1 - l0) / l1
    u = (s - s0*(th.sign(s))) * (th.abs(s) > s0).float()
    return L2_reconstr_loss(A, u, x, sigma) + l1 * th.norm(s, p=1)
class MixT_SDE:
    def __init__(self, params, E):
        self.params = params
        self.E = E
    def zero_grad(self):
        for _, p in self.params.items():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
    def step(self, dt):
        update = {}
        self.zero_grad()
        self.E(self.params).backward()
        for n, p in self.params.items():
            if p.grad is None:
                continue
            dtau = dt * p.freq

            dEdp = p.grad * dtau
            dW = random.gauss(0, dtau**0.5)
            if p.T > 0:
                update[n] = -dEdp + dW * p.T
            else:
                update[n] = -dEdp

        for n, dp in update.items():
            self.params[n].data += dp

if __name__ == '__main__':
    tau_s = 1e-2
    tau_x = 1
    tau_A = 4e1
    T_RANGE = 1e2
    T_STEPS = int(1e4)

    l1 = .5
    l0 = .8
    sigma = 1.

    N_DIM = 2
    N_SPARSE = 3

    A = get_param((N_DIM, N_SPARSE), tau=tau_A, T=0)
    s = get_param((N_SPARSE), tau=tau_s)
    x = get_param((N_DIM), tau=tau_x)
    l0 = get_param(1, tau=None, init=l0)
    l1 = get_param(1, tau=None, init=l1)
    sigma = get_param(1, tau=None, init=sigma)

    params = {
            'A' : A,
            's' : s,
            'x' : x,
            'l0' : l0,
            'l1' : l1,
            'sigma' : sigma
            }

    mt_sde = MixT_SDE(params, E)
    for _ in range(10):
        print('============')
        print(mt_sde.params)
        mt_sde.step(.1)

