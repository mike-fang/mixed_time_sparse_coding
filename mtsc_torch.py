import torch as th
from torch.nn import Parameter
from torch.nn import Module
import torch.nn.functional as F
import numpy as np
from loaders import StarLoader, Loader
from tqdm import tqdm
import matplotlib.pylab as plt
from matplotlib import animation
from visualization import show_2d_evo
from solution_saver import Solutions, get_tmp_path
import json

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
                self.data += self.temp * dW / (dt * self.tau)**0.5
        else:
            self.data += 0.5 * self.momentum * dt / self.mass
    def update_p(self, dt):
        if self.mu == 0:
            return 
        self.momentum += -self.tau * self.momentum * dt / self.mass - self.grad * dt
        if self.temp > 0:
            dW = th.FloatTensor(self.size()).normal_()
            self.momentum += (self.temp * self.tau / dt)**0.5 * dW
     
class MixedTimeSC(Module):
    @classmethod
    def from_json(cls, f_name=None):
        if f_name is None:
            f_name = get_tmp_path(load=True, f_name = 'hyper_params.json')
        with open(f_name, 'r') as f:
            hp = json.load(f)
        return cls(**hp)
    def __init__(self, n_dim, n_dict, n_batch, tau={}, mass={}, T={}, positive=False):
        super().__init__()
        self.n_dim = n_dim
        self.n_dict = n_dict
        self.n_batch = n_batch

        self.tau = tau
        self.mass = mass
        self.T = T
        self.positive = positive

        self.init_params()
    def init_params(self, init=[None]):
        self.A = MTParameter(th.Tensor(self.n_dim, self.n_dict))
        self.s = MTParameter(th.Tensor(self.n_dict, self.n_batch))
        self.rho = MTParameter(th.Tensor(1))
        self.l1 = MTParameter(th.Tensor(1))
        self.nu = MTParameter(th.Tensor(1))

        # nu = log(pi / (1-pi)) log odds ratio
        self.reset_params(init=init)
        self.set_properties()
    def reset_params(self, init=[None]):
        self.A.data.normal_()
        self.s.data.normal_()
        #self.s.data *= 0
        self.rho.data = th.tensor(1.)
        self.l1.data = th.tensor(1.)
        self.nu.data = th.tensor(1.)
        for n, p in self.named_parameters():
            if n in init:
                p.data = th.tensor(init[n], dtype=th.float)
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
        if self.positive:
            beta = 1
            u = F.softplus(beta * u) 
        return u
    @property
    def r(self):
        r = self.A @ self.u
        return r.T
    def recon_error(self, x):
        return 0.5 *  ((x - self.r)**2).sum()
    def sparse_loss(self):
        return th.abs(self.s).sum() / self.n_batch
    def energy(self, x):
        return self.rho * self.recon_error(x) + self.l1 * self.sparse_loss()
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
        elif isinstance(out_t, int):
            skip = int(len(tspan) / out_t)
            out_t = tspan[::skip]
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
            soln[n] = np.zeros((len(out_t), *p.size()))
        soln['u'] = np.zeros((len(out_t), *self.s.size()))
        soln['x'] = np.zeros((len(out_t), self.n_batch, self.n_dim))
        soln['r'] = np.zeros((len(out_t), self.n_batch, self.n_dim))
        soln['T'] = np.zeros(len(out_t))

        def add_to_soln(i):
            for n, p in self.named_parameters():
                soln[n][i] = p.data
            soln['u'][i] = self.u.data
            soln['x'][i] = x.data
            soln['r'][i] = self.r.data
            soln['T'][i] = out_t[i]

        add_to_soln(0)
        soln_counter = 1
        for n, t in enumerate(tqdm(tspan[1:])):
            if new_batch[n]:
                x = loader()
            dt = dT[n]
            self.update_params(x, dt)

            #Store solution if in out_t
            if t == out_t[soln_counter]:
                add_to_soln(soln_counter)
                soln_counter += 1

        return soln
    @property
    def hyper_params(self):
        hyper_params = {
                k : self.__dict__[k] for k in self.__dict__ if k in ['n_dim', 'n_dict', 'n_batch', 'tau', 'mass', 'T', 'positive']
                }
        return hyper_params
    def save_hyper_params(self, f_name=None):
        if f_name is None:
            f_name = get_tmp_path(f_name = 'hyper_params.json')
        with open(f_name, 'w') as f:
            json.dump(self.hyper_params, f)
        return self.hyper_params
    def __repr__(self):
        str_ = ('Mixed Time Sparse Coding Model with L0 Energy\n')
        for k, v in self.hyper_params.items():
            str_ += f'\t{k} : {v} \n'
        return str_

if __name__ == '__main__':
    n_dict = 3
    n_dim = 2
    n_batch = 3

    mass = {
            's' : .00,
            'A' : 0,
            }
    T = {
            's': 1.,
            }
    tau = {
            's': 1e2,
            'x': 5e2,
            'A': 1e4,
            'l1': None,
            'nu': None,
            'rho': None,
            }

    init = {
            'l1' : 1.,
            'nu' : .5,
            'rho' : 1.,
            }

    T_RANGE = 1e3
    T_STEPS = int(T_RANGE)
    tspan = np.linspace(0, T_RANGE, T_STEPS, endpoint=False)
    loader = StarLoader(n_basis=3, n_batch=n_batch)

    mtsc = MixedTimeSC(n_dim, n_dim, n_batch, tau=tau, mass=mass, T=T, positive=True)
    mtsc = MixedTimeSC(n_dim, n_dict, n_batch, tau=tau, mass=mass, T=T, positive=True)
    mtsc.reset_params(init=init)
    soln_dict = mtsc.train(loader, tspan, None)
    soln = Solutions(soln_dict)
    show_2d_evo(soln)
    assert False

    try:
        mtsc = MixedTimeSC.from_json()
        mtsc.load_state_dict(th.load(get_tmp_path(load=True, f_name='state_dict.pth')))
    except:
        mtsc = MixedTimeSC(n_dim, n_dict, n_batch, tau=tau, mass=mass, T=T, positive=True)
        mtsc.reset_params(init=init)
        th.save(mtsc.state_dict(), get_tmp_path(load=True, f_name='state_dict.pth'))

    try:
        soln = Solutions.load()
    except:
        soln_dict = mtsc.train(loader, tspan, None)
        soln = Solutions(soln_dict)
    show_2d_evo(soln)
