import torch as th
from torch.optim.optimizer import Optimizer, required
from collections import defaultdict 

class EulerMaruyama(Optimizer):
    def __init__(self,  param_groups, dt=1, tau=required, mu=0, T=0, noise_chache=0):
        for n, d in enumerate(param_groups):
            if d['tau'] in [None, 0]:
                param_groups.pop(n)
        defaults = dict(dt=dt, tau=tau, mu=mu, T=T)
        super().__init__(param_groups, defaults)
        self.noise_chache = noise_chache
        if noise_chache:
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
                param_state['noise'] = th.FloatTensor(self.noise_chache, *p.shape).normal_(0, scale)
    def reset_noise(self):
        for p, param_state in self.state.items():
            if 'noise_scale' in param_state:
                self.noise_idx = 0
                scale = param_state['noise_scale']
                param_state['noise'].normal_(0, scale)
    def step(self, closure):
        # Half-step x if there is mass
        for group in self.param_groups:
            mu = group['mu']
            if mu == 0:
                continue
            tau = group['tau']
            du =  group['dt'] / tau

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
            du =  group['dt'] / tau

            for p in group['params']:
                p_grad = p.grad
                if self.noise_chache:
                    eta = self.state[p]['noise'][self.noise_idx]
                else:
                    eta = th.FloatTensor(p.shape).normal_()

                if mu > 0:
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
                        p.data.add_((2 * T * du)**0.5, eta)

        if self.noise_chache:
            self.noise_idx += 1
            if self.noise_idx >= self.noise_chache:
                self.reset_noise()
        return energy

if __name__ == '__main__':
    from torch.nn import Parameter
    import matplotlib.pylab as plt
    import numpy as np
    from time import time
    steps = 10000
    noise_chache = 10000

    x = Parameter(th.tensor(0.))
    y = Parameter(th.tensor(0.))
    def closure():
        energy = (x**2)/2 + (y**2)/2
        energy.backward()
        return energy

    param_groups = [
            {'params': [x], 'tau': 1, 'mu': 10, 'T': 1},
            {'params': [y], 'tau': 1, 'mu': 10, 'T': 1},
            ]
    optimizer = EulerMaruyama(param_groups, dt=1, noise_chache=noise_chache)


    X = []
    Y = []
    t0 = time()
    for _ in range(steps):
        optimizer.zero_grad()
        optimizer.step(closure)
        X.append(float(x))
        Y.append(float(y))
    print(time() - t0)

    plt.subplot(211)
    plt.plot(X)
    plt.plot(Y)
    plt.subplot(212)
    X_range = np.linspace(-3, 3, 100)
    p = np.exp(-X_range**2/2)
    p /= p.sum() * (X_range[1] - X_range[0])
    
    plt.hist(X, bins=50, density=True)
    plt.plot(X_range, p)
    plt.show()
