import torch as th
from torch.optim.optimizer import Optimizer, required
from collections import defaultdict 
import numpy as np


class EulerMaruyama(Optimizer):
    def __init__(self,  param_groups, tau=required, mu=0, T=0, coupling=1, noise_cache=0):
        for n, d in enumerate(param_groups):
            tau = d['tau']
            if tau in [None, 0]:
                param_groups.pop(n)
        defaults = dict(tau=tau, mu=mu, coupling=coupling, T=T)
        super().__init__(param_groups, defaults)
        self.noise_cache = noise_cache
        if noise_cache:
            self.init_noise()
            self.noise_idx = 0
    def init_noise(self):
        for group in self.param_groups:
            scale = (2 * group['T'] * group['dt'] / group['tau'])**0.5
            if scale == 0:
                continue
            mu = group['mu']
            for p in group['params']:
                param_state = self.state[p]
                param_state['noise_scale'] = scale
                param_state['noise'] = th.FloatTensor(self.noise_cache, *p.shape).normal_(0, scale)
    def reset_noise(self, noise_cache=None):
        self.noise_idx = 0
        if noise_cache is not None:
            self.noise_cache = noise_cache
            self.init_noise()
            return
        for p, param_state in self.state.items():
            if 'noise_scale' in param_state:
                scale = param_state['noise_scale']
                param_state['noise'].normal_(0, scale)
    def step(self, closure, dt=1):
        # Half-step x if there is mass
        for group in self.param_groups:
            mu = group['mu']
            if mu == 0:
                continue
            tau = group['tau']
            du =  group['coupling'] * dt / tau


            for p in group['params']:
                param_state = self.state[p]
                if 'momentum' not in param_state:
                    param_state['momentum'] = th.zeros_like(p)
                pi = param_state['momentum']
                p.data.add_(du/(2 * mu), pi)

        energy = closure() 

        for group in self.param_groups:
            mu = group['mu']
            tau = group['tau']
            T = group['T']
            du =  group['coupling'] * dt / tau

            if group['params'][0].shape == (16, 8):
            #    print(du)
                pass
            if du == 0:
                continue

            for p in group['params']:
                p_grad = p.grad
                if T > 0:
                    if self.noise_cache:
                        eta = self.state[p]['noise'][self.noise_idx]
                    else:
                        eta = th.FloatTensor(p.shape).normal_()
                    if p.is_cuda:
                        eta = eta.to('cuda')

                if mu != 0:
                    # Step momentum if there is mass
                    pi = self.state[p]['momentum']
                    pi.add_(-du, pi/mu + p_grad)
                    
                    # Add noise if there is temperature
                    if T > 0:
                        pi.add_((2 * T * du)**0.5, eta)

                    # Half-step x
                    p.data.add_(du/(2 * mu), pi)
                else:
                    # If no mass, just step x
                    p.data.add_(-du, p_grad)
                    if T > 0:
                        p.data.add_((np.abs(2 * T * du))**0.5, eta)

        if self.noise_cache:
            self.noise_idx += 1
            if self.noise_idx >= self.noise_cache:
                self.reset_noise()
        return energy

if __name__ == '__main__':
    from torch.nn import Parameter
    import matplotlib.pylab as plt
    import numpy as np
    from time import time
    from tqdm import tqdm
    steps = 300000
    noise_cache = 0


    tau_x = -1
    tau_t = 100
    tau_mu = 30
    SIMGA = 1
    MU0 = 0
    dt = 1
    scale = int(100/dt)
    x = Parameter(th.tensor(1.))
    x0 = Parameter(th.tensor(0.))
    t = Parameter(th.tensor(-3.))
    mu = Parameter(th.tensor(3.))
    print(th.exp(-t/2))
    def energy(x, t, mu):
        return 0.5 * th.exp(t) * (x-mu)**2
    def next_batch():
        x0.data.normal_(0, SIMGA)
    def closure():
        E_eff = 0
        E_eff += energy(x0, t, mu) 
        E_eff -= energy(x, t, mu)
        E_eff.backward()
        return E_eff

    param_groups = [
            {'params': [x], 'tau': tau_x, 'T': 1.000},
            {'params': [t], 'tau': tau_t, 'mu':.5, 'T': 0},
            {'params': [mu], 'tau': tau_mu, 'mu':.5, 'T': 0},
            ]
    solver = EulerMaruyama(param_groups, dt=.01)

    X = np.zeros(steps)
    X0 = np.zeros(steps)
    T = np.zeros(steps)
    MU0 = np.zeros(steps)
    MU = np.zeros(steps)
    for n in tqdm(range(steps)):
        if n % (abs(tau_x)*scale) == 0:
            next_batch()
        solver.zero_grad()
        solver.step(closure)
        X[n] = float(x)
        X0[n] = float(x0)
        T[n] = float(t)
        MU[n] = float(mu)

    plt.plot(np.ones(steps)*SIMGA, 'b--')
    plt.plot(-np.ones(steps)*SIMGA, 'b--')
    plt.plot(X0, 'k', label = 'X ~ data')
    plt.plot(X[X**2 < 1e5], 'r', label = 'X ~ model')
    plt.plot(MU + np.exp(-T/2), 'g--', label = 'Scale')
    plt.plot(MU -np.exp(-T/2), 'g--', label = 'Scale')
    plt.plot(MU, 'g-', label = 'Mean')
    plt.legend()
    plt.show()
