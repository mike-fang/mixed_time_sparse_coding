import torch as th
from torch.nn import Parameter
from torch.nn import Module
import torch.nn.functional as F
import numpy as np
from loaders import StarLoader
from tqdm import tqdm
import matplotlib.pylab as plt

class MTParameter(Parameter):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mu = 0
        self.tau = None
        self.temp = 0
        self.momentum = th.zeros(self.shape)
    @property
    def mass(self):
        return self.mu * self.tau**2
    def update_x(self, dt):
        if self.tau in [0, None]:
        # Fixed parameter
            return
        if self.mu == 0:
        # No mass
            self.data += -self.grad/self.tau * dt
            if self.temp > 0:
                dW = th.FloatTensor(self.size()).normal_()
                self.data += self.temp * dW / self.tau**0.5
        else:
            self.data += 0.5 * self.momentum * dt / self.mass
    def update_p(self, dt):
        if self.mu == 0:
            return 
        self.momentum += -self.tau * self.momentum * dt / self.mass - self.grad
        if self.temp > 0:
            dW = th.FloatTensor(self.size()).normal_()
            self.momentum += (self.temp * self.tau)**0.5 * dW
     
class MixedTimeSC(Module):
    def __init__(self, n_dim, n_dict, n_batch, tau, mass, T, positive=False):
        super().__init__()
        self.n_dim = n_dim
        self.n_dict = n_dict
        self.n_batch = n_batch

        self.tau = tau
        self.mass = mass
        self.T = T
        self.positive = positive

        self.init_params()
        self.set_properties()
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
    def set_properties(self):
        for n, p in self.named_parameters():
            if n in self.tau:
                p.tau = self.tau[n]
                if n in self.mass:
                    p.mu = self.mass[n]
                if n in self.T:
                    p.temp = self.T[n]
    @property
    def u(self):
        s0 = -self.l1 * th.log(th.sigmoid(-self.nu))
        u = F.relu(self.s - s0) - F.relu(-self.s - s0)
        return u
    def energy(self, x):
        A = self.A
        s0 = -self.l1 * th.log(th.sigmoid(-self.nu))
        u = F.relu(self.s - s0) - F.relu(-self.s - s0)
        return (self.rho * ((x.T - A @ u)**2).sum() + self.l1 * th.abs(self.s).sum()) / self.n_batch
    def update_params(self, x, dt):
        for n, p in self.named_parameters():
            if p.mu > 0:
                # Half step x
                p.update_x(dt/2)
        self.zero_grad()
        self.energy(x).backward()
        for n, p in self.named_parameters():
            if p.mu > 0:
                # Update p
                p.update_p(dt)
                # Half step x
                p.update_x(dt/2)
            else:
                p.update_x(dt)
    def train(self, loader, tspan, out_t=None):
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
        x_idx = (tspan // self.tau['x']).astype(int)
        new_batch = (x_idx[1:] - x_idx[:-1]).astype(bool)
        x = loader()
        
        soln = {}
        for n, p in self.named_parameters():
            soln[n] = th.Tensor(len(out_t), *p.size())
        soln['u'] = th.Tensor(len(out_t), *self.s.size())
        soln['x'] = th.Tensor(len(out_t), self.n_dict, self.n_dim)

        soln_counter = 0
        for n, dt in enumerate(tqdm(dT)):
            t = tspan[n]
            self.update_params(x, dt)

            #Store solution if in out_t
            if t == out_t[soln_counter]:
                for n, p in self.named_parameters():
                    soln[n][soln_counter] = p
                soln['u'][soln_counter] = self.u
                soln['x'][soln_counter] = x

        return soln


if __name__ == '__main__':
    n_dict = 3
    n_dim = 2
    n_batch = 3
    mass = {
            's' : .05,
            }
    T = {
            's': 1,
            }
    tau = {
            's': 1e2,
            'x': 1e3,
            'A': 1e5,
            'l1': None,
            'nu': None,
            'rho': None,
            }

    T_RANGE = 1e3
    T_STEPS = int(T_RANGE)
    tspan = np.linspace(0, T_RANGE, T_STEPS, endpoint=False)

    mtsc = MixedTimeSC(n_dim, n_dict, n_batch, tau, mass, T)
    loader = StarLoader(3, n_batch)
    soln = mtsc.train(loader, tspan, None)
    R = th.einsum('ijk,ilk->ijl', soln['u'], soln['A'])
    plt.scatter(*R[:, 0, :].data.numpy().T)
    plt.scatter(*R[:, 1, :].data.numpy().T)
    plt.scatter(*R[:, 2, :].data.numpy().T)
    plt.show()
