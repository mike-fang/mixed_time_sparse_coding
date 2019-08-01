import torch as th
from torch.nn.parameter import Parameter
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt
from matplotlib import animation
import os.path as osp
from time import time
import random
def L2_reconstr_loss(A, s, x, sigma):
    print(f'x : {x}')
    print(f'A@s : {A@s}')
    return 0.5 * th.norm(x - A @ s)**2 / sigma**2

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
def shuffle_param(param, freq, tspan):
    #freq = param.freq
    idx = (tspan * freq).long()
    max_idx = idx.max()
    n_data = len(param)
    rand_idx = th.zeros(max_idx + 1).long().random_(n_data)
    return param[rand_idx[idx]]
class MixT_SDE:
    def __init__(self, params, E):
        self.params = params
        self.E = E
    def zero_grad(self):
        for _, p in self.params.items():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
    def step(self, t, dt, dW=None):
        energy = self.E(self.params, t)
        self.zero_grad()
        energy.backward()
        print('energy : ', energy)
        for n, p in self.params.items():
            if p.grad is None:
                continue
            dtau = dt * p.freq
            dEdp = p.grad * dtau

            print('name: ', n)
            print('param: ', p.data)
            print('dEdp: ', dEdp.data)
            if p.T > 0:
                dW = th.zeros_like(p)
                dW.normal_()
                dW *= (2 * dtau)**0.5 
                print('dtau: ', dtau)
                print('dW: ', dW.data)
                update[n] = -dEdp + dW * p.T 
            else:
                update[n] = -dEdp
            self.params[n]
            #print(update)
            print('--------')

        for n, dp in update.items():
            self.params[n].data += dp
    def solve(self, solve_name, X, tspan):
        solve_param = self.params[solve_name]
        solve_freq = solve_param.freq

        #Freeze solve_param evolution, it is to be driven 
        solve_param.requires_grad = False
        shuffled_X = shuffle_param(X, solve_freq, tspan)
        solve_param.data = shuffled_X[0]

        # Get time_differences
        t_steps = len(tspan)
        dt = tspan[1:] - tspan[:-1]
        dW = np.random.random() * dt**0.5

        # Initialize param evolution with t=0 values
        param_evol = {}
        for n, p in self.params.items():
            if not p.requires_grad:
                pass
                #continue
            evol = th.zeros((t_steps, *p.shape))
            evol[0] = p.data
            param_evol[n] = evol

        for i, t in enumerate(tspan[1:]):
            solve_param.data = shuffled_X[i+1]
            self.step(t, dt[i], None)
            for name, ev in param_evol.items():
                ev[i+1] = self.params[name].data
        param_evol['tspan'] = tspan
        return param_evol


if __name__ == '__main__':
    n_dim = n_sparse = 2

    tau_s = 1e-2
    tau_x = 1
    tau_A = 4e1
    
    A = get_param((n_dim, n_sparse), tau=None, T=0)
    s = get_param(n_sparse, tau=tau_s, T=0)
    x = get_param(n_dim, tau=None, T=0)
    A.data = th.eye(2)
    x.data *= 0
    s.data = th.Tensor((100, 0))


    l0 = get_param(1, tau=None, init=.5)
    l1 = get_param(1, tau=None, init=.8)
    sigma = get_param(1, tau=None, init=1)
    params = {
            'A' : A,
            's' : s,
            'x' : x,
            'l0' : l0,
            'l1' : l1,
            'sigma' : sigma
            }

    def E(params, t):
        A = params['A']
        s = params['s']
        x = params['x']
        sigma = params['sigma']
        return L2_reconstr_loss(A, s, x, sigma)
    mtsde = MixT_SDE(params, E)
    for _ in range(10):
        mtsde.step(0, 1e-3)
