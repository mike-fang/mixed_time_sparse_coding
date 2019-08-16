import torch as th
from torch.nn import Parameter
from torch.nn import Module
import torch.nn.functional as F
import numpy as np

class MTParameter(Parameter):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mu = 0
        self.tau = None
        self.momentum = th.zeros(self.shape)
    @property
    def mass(self):
        return self.mu * self.tau**2
     
def update_x(param, dt, dW=None):
    if param.tau == 0:
        return
    if param.mu == 0:
        param.data += -param.grad/param.tau * dt
        if dW is not None:
            param.data += dW / param.tau**0.5
    else:
        param.data += param.momentum * dt / param.mass
def update_p(param, dt, dW=None):
    param.momentum += -param.tau * param.momentum * dt / param.mass - param.grad
    if dW is not None:
        param.momentum += param.tau**0.5 * dW

class MixedTimeSC(Module):
    def __init__(self, n_dim, n_dict, n_batch, tau, mass, positive=False):
        super().__init__()
        self.n_dim = n_dim
        self.n_dict = n_dict
        self.n_batch = n_batch
        self.tau = tau
        self.mass = mass
        self.positive = positive

        self.init_params()
        self.set_tau_mass()
    def init_params(self):
        self.A = MTParameter(th.Tensor(self.n_dim, self.n_dict))
        self.s = MTParameter(th.Tensor(self.n_dict, self.n_batch))
        self.rho = MTParameter(th.Tensor(1))
        self.l1 = MTParameter(th.Tensor(1))
        self.nu = MTParameter(th.Tensor(1))
        # nu = log(pi / (1-pi)) log odds ratio
        self.reset_params()
    def reset_params(self):
        self.A.data.normal_()
        self.s.data.normal_()
        #self.s.data *= 0
        self.rho.data = th.tensor(1.)
        self.l1.data = th.tensor(1.)
        self.nu.data = th.tensor(1.)
    def set_tau_mass(self):
        for n, p in self.named_parameters():
            if n in self.tau:
                p.tau = self.tau[n]
                if n in self.mass:
                    p.mu = self.mass[n]
    def energy(self, x):
        A = self.A
        s0 = -self.l1 * th.log(F.sigmoid(-self.nu))
        u = F.relu(self.s - s0) - F.relu(-self.s - s0)
        return (self.rho * ((x - A @ u)**2).sum() + self.l1 * th.abs(self.s).sum()) / self.n_batch
    def __call__(self, x):
        return self.energy(x)

class MTSCOptimizer:
    def __init__(self, model):
        self.model = model
    def train(self, loader, tspan):
        # Start tspan at 0
        tspan -= tspan.min()

        # If not specified, output all time points
        if out_t is None:
            out_t = tspan
        n_out = len(out_t)

        # Definite dt, t_steps
        t_steps = len(tspan)
        dT = tspan[1:] - tspan[:-1]
        
        # Find where to load next X batch
        x_idx = (tspan // self.params['tau_x']).astype(int)
        new_batch = (x_idx[1:] - x_idx[:-1]).astype(bool)


    def update_params(self, dt, dW=None):
        for p in self.parameters():
            if p.mu != 0:
                # Half step x
                update_x(p, dt/2)
        self.energy


def update_param(p, dt, tau, mu, T=1, dW=None):
    if dW is None:
        T = 0
    if mu == 0:
        # No mass
        dx = -dEdx / tau
        if T > 0:
            dx += (T / tau)**0.5 * dW
        return dx, 0
    else:
        # Mass
        m = mu * tau**2
        dx = p * dt / (2*m)
        dp = -tau * p * dt / m - dEdx
        if T > 0:
            dp += (T * tau)**0.5 * dW
        dx += p * dt / (2*m)
        return dx, dp


if __name__ == '__main__':
    n_dict = 3
    n_dim = 2
    n_batch = 3
    mass = {
            's' : .05,
            }
    tau = {
            's': 1e2,
            'x': 1e3,
            'A': 1e5,
            'l1': 1e5,
            'nu': 1e5,
            'rho': 1e5,
            }

    X = th.zeros(n_dict, n_batch)
    X.normal_()
    X = Parameter(X)

        
    mtsc = MixedTimeSC(n_dict, n_dict, n_batch, tau, mass)
    #mtsc.state_dict()['A'].mu = 1
    E = mtsc(X)
    E.backward()
    for n, p in mtsc.named_parameters():
        print(p.grad)

