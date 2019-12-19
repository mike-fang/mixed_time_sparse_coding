import numpy as np
import torch as th
from torch.nn import Parameter
from torch.nn import Module
import torch.nn.functional as F
from euler_maruyama import EulerMaruyama
from tqdm import tqdm
from loaders import StarLoader, Loader, StarLoader_

class DiscreteSCModel(Module):
    def __init__(self, n_basis, n_dim, n_batch, l1=1, pi=.5, positive=False):
        super().__init__()
        self.n_basis = n_basis
        self.n_dim = n_dim
        self.n_batch = n_batch
        self.l1 = l1
        self.pi = pi
        self.u0 = -np.log(pi) / l1
        self.init_params()
        self.positive = positive
    def init_params(self):
        self.A = Parameter(th.Tensor(self.n_dim, self.n_basis))
        self.u = Parameter(th.Tensor(self.n_basis, self.n_batch))
        self.x = Parameter(th.Tensor(self.n_dim, self.n_batch))
        self.reset_params()
    def reset_params(self, setval=None):
        for p in self.parameters():
            if setval is None:
                p.data.normal_()
            else:
                if p in setval:
                    if setval[p] is None:
                        p.data.normal_()
                    else:
                        p.data = setval[p]
    def get_recon(self, u):
        if self.positive:
            s = F.relu(u - self.u0) + F.relu(-u - self.u0)
        else:
            s = F.relu(u - self.u0) - F.relu(-u - self.u0)
        return (self.A @ s)
    def energy(self, u=None, x=None):
        if x is None:
            x = self.x
        if u is None:
            u = self.u
        r = self.get_recon(u)
        recon_loss = 0.5 * ((r - x)**2).sum()
        sparse_loss = th.abs(r).sum()
        return recon_loss + self.l1 * sparse_loss
    def forward(self, x):
        return self.energy(x=x)

if __name__ == '__main__':
    ITER_A = 1
    ITER_U = 10
    N_BATCH = 3
    N_BASIS = 3
    LR_U = .01
    LR_A = .1
    dsc = DiscreteSCModel(n_dim=2, n_basis=N_BASIS, n_batch=N_BATCH, pi=.5)
    loader = StarLoader(N_BASIS, N_BATCH, sigma=0, pi=0.5, l1=1)
    param_groups = [
            {'params': [dsc.u], 'tau':LR_U**(-1)},
            {'params': [dsc.A], 'tau':0},
            ]
    solver = EulerMaruyama(param_groups, dt=1)


    for n_a in tqdm(range(ITER_A)):
        X = loader(transposed=False)
        def closure():
            E = dsc.energy(x=X)
            E.backward()
            return E
        solver = EulerMaruyama(param_groups)
        dsc.reset_params({dsc.u : None})
        for n_u in tqdm(range(ITER_U)):
            solver.step(closure)
