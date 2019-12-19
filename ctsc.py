from euler_maruyama import EulerMaruyama
import torch as th
from torch.nn import Parameter, Module
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

def get_dt(tspan):
    dt = np.zeros_like(tspan)
    dt[1:] = tspan[1:] - tspan[:-1]
    return dt
def get_time_interval(tspan, N=None, dt=None):
    tspan -= min(tspan)
    if N is not None:
        dt = tspan[-1]/N

    interval = np.zeros_like(tspan, dtype=bool)

    for n in np.unique(tspan//dt):
        idx = np.argmin(np.abs(tspan - n*dt))
        interval[idx] = True

    return interval

class CTSCModel(Module):
    # // Init //
    def __init__(self, n_dim, n_dict, n_batch, positive=False, sigma=1, l1=1, pi=0.5):
        super().__init__()
        self.n_dim = n_dim
        self.n_dict = n_dict
        self.n_batch = n_batch

        self.positive = positive
        self.sigma= sigma
        self.l1 = l1
        self.pi = pi
        self.init_params()
    def init_params(self):
        self.A = Parameter(th.Tensor(self.n_dim, self.n_dict))
        self.u = Parameter(th.Tensor(self.n_dict, self.n_batch))
        self.reset_params()
    def reset_params(self):
        self.A.data.normal_()
        self.u.data.normal_()
    @property
    def pi(self):
        return self._pi
    @pi.setter
    def pi(self, pi):
        self._pi = pi
        self.u0 = - np.log(pi) / self.l1
    @property
    def s(self):
        if self.positive:
            s = F.relu(self.u - self.u0) + F.relu(-self.u - self.u0)
        else:
            s = F.relu(self.u - self.u0) - F.relu(-self.u - self.u0)
        return s
    @property
    def r(self):
        return (self.A @ self.s).t()
    def energy(self, x):
        recon = 0.5 * ((self.r - x)**2).sum()
        sparse = self.l1 * th.abs(self.u).sum()
        return recon/self.sigma**2 + self.l1 * sparse
        return 0.5 * ((self.r - x)**2).sum()/self.sigma**2 + self.l1 * th.abs(self.u).sum()
    def forward(self, x):
        return self.energy(x)

class CTSCSolver:
    def __init__(self, model, loader, tau_A, tau_u, tau_x, mu_A=0., mu_u=0., T_A=0., T_u=1., asynch=False):
        self.model = model
        self.n_batch = model.n_batch

        self.loader = loader
        self.tau_x = tau_x
        optim_params = [
                {'params':[model.A], 'tau':tau_A, 'mu':mu_A, 'T':T_A},
                {'params':[model.u], 'tau':tau_u, 'mu':mu_u, 'T':T_u},
                ]
        self.optimizer = EulerMaruyama(optim_params)
        self.asynch = asynch
    def load_batch_size(self, tspan):
        where_load = th.zeros(len(tspan), self.n_batch)
        if self.asynch:
            dt = th.tensor(get_dt(tspan))
            p = (dt/self.tau_x)[:, None]
            where_load.bernoulli_(p)
        else:
            for n in np.unique(tspan // self.tau_x):
                idx = np.argmin(np.abs(tspan - n*self.tau_x))
                where_load[idx] = self.model.n_batch
        return where_load
    def solve(self, tspan, normalize_A=True, save_N=None, save_T=None):
        # Get initial data x
        x = self.loader()

        # Define closure for solver
        def closure():
            energy = self.model(x)
            energy.backward()
            return energy
  
        # When to load new data
        load_n = self.load_batch_size(tspan)

        # Where to save output
        if save_N is not None:
            save_time = get_time_interval(tspan, N=save_N)
        elif save_T is not None:
            save_time = get_time_interval(tspan, dt=save_T)
        else:
            save_time = None

        # Delta Ts in case dt != 1
        dtspan = get_dt(tspan)

        soln = defaultdict(list)

        # Iterate over tspan
        for n, t in enumerate(tqdm(tspan)):
            dt = dtspan[n]

            # Load new batch asynchronously if necessary
            batch_size = int(load_n[n].sum())
            if batch_size > 0:
                #TODO: remove this stupid switcharoo
                x_ = x.clone()
                x_.data[load_n[n].nonzero()] = loader(n_batch=batch_size).view(batch_size, 1, -1)
                x = x_

            # Optimizer step
            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            # Normalize dictionary if needed
            if normalize_A:
                self.model.A.data = self.model.A / self.model.A.norm(dim=0)


            # Save solution
            if save_time is not None and save_time[n]:
                soln['r'].append(self.model.r.data.numpy())
                soln['u'].append(self.model.u.data.numpy())
                soln['s'].append(self.model.s.data.numpy())
                soln['A'].append(self.model.A.data.numpy())
                soln['x'].append(x.data.numpy())
                soln['t'].append(t)
        if soln:
            for k, v in soln.items():
                soln[k] = np.array(v)
            return soln
        else:
            return None
    def save_checkpoint(self, t):
        checkpoint = {
                't': t,
                'model_sd': self.model.state_dict(),
                'solver_sd': self.optimizer.state_dict(),
                }

        path = os.path.join(out_dir, f'checkpoint_{t}.pth')
        th.save(checkpoint, path)
        print(f'Checkpoint saved to {path}')
        return checkpoint


if __name__ == '__main__':
    from loaders import BarsLoader
    from visualization import show_img_XRA

    H = W = 3
    N_BATCH = H + W
    PI = 0.5
    loader = BarsLoader(H, W, N_BATCH, p=PI)

    model_params = dict(
            n_dict = H * W,
            n_dim = H * W,
            n_batch = N_BATCH,
            positive=True,
            )
    solver_params = dict(
            tau_A = 5e3,
            tau_u = 1e2,
            tau_x = 1e3,
            asynch=True,
            )

    model = CTSCModel(**model_params)
    solver = CTSCSolver(model, loader, **solver_params)

    tspan = np.arange(1e3)

    soln = solver.solve(tspan, save_N=1e2)
    X = soln['x']
    R = soln['r']
    A = soln['A']

    show_img_XRA(X, R, A, img_shape=(H, W))

