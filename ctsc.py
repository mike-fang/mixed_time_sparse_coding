from euler_maruyama import EulerMaruyama
import torch as th
from torch.nn import Parameter, Module
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
import h5py
import yaml
from time import time
import os.path
from glob import glob

FILE_DIR = os.path.abspath(os.path.dirname(__file__))

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
def get_timestamped_dir(load=False, base_dir=None):
    if base_dir is None:
        base_dir = 'tmp'
    if load:
        tmp_files = glob(os.path.join(FILE_DIR, 'results', base_dir, 'exp_*'))
        tmp_files.sort()
        dir_name = tmp_files[-1]
    else:
        time_stamp = f'exp_{time():.0f}'
        dir_name = os.path.join(FILE_DIR, 'results', base_dir, time_stamp)
        os.makedirs(dir_name)
    return dir_name
def load_solver(dir_path, loader):
    with open(os.path.join(dir_path, 'params.json'), 'r') as f:
        params = yaml.load(f)
    model_params = params['model_params']
    solver_params = params['solver_params']
    model = CTSCModel(**model_params)
    solver = CTSCSolver(model, loader, **solver_params)
    return solver

class CTSCModel(Module):
    # // Init //
    def __init__(self, n_dim, n_dict, n_batch, positive=False, sigma=1, l1=1, pi=0.5):
        # Record params
        self.params = {k:v for (k, v) in locals().items() if isinstance(v, (int, float, bool))}
        super().__init__()
        self.n_dim = n_dim
        self.n_dict = n_dict
        self.n_batch = n_batch

        self.positive = positive
        self.sigma= sigma
        self.l1 = l1
        self.pi = pi
        self.init_params()
    def hyperparams(self):
        hp = dict(
                n_dim=self.n_dim,
                n_dict=self.n_dict,
                n_batch=self.n_batch,
                positive=self.positive,
                sigma=self.sigma,
                l1=self.l1,
                pi=self.pi,
                )
        return hp
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
        # Record params
        self.params = {k:v for (k, v) in locals().items() if isinstance(v, (int, float, bool))}

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
    def solve(self, tspan, normalize_A=True, out_N=None, out_T=None, save_soln=False):
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
        if out_N is not None:
            save_time = get_time_interval(tspan, N=out_N)
        elif out_T is not None:
            save_time = get_time_interval(tspan, dt=out_T)
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
    def save_hyperparams(self, dir_path):
        params = dict(
                model_params=self.model.params,
                solver_params=self.params,
                )
        with open(os.path.join(dir_path, 'params.json'), 'w') as f:
            yaml.dump(params, f)
    def save_soln(self, soln, base_dir):
        t_last = soln['t'][-1]
        dir_path = get_timestamped_dir(base_dir=base_dir)
        self.save_hyperparams(dir_path)
        self.save_checkpoint(dir_path, t_last)
        soln_h5 = h5py.File(os.path.join(dir_path, 'soln.h5'), 'w')
        for k, v in soln.items():
            soln_h5.create_dataset(k, data=v)
    def save_checkpoint(self, out_dir, t):
        checkpoint = {
                't': t,
                'model_sd': self.model.state_dict(),
                'optim_sd': self.optimizer.state_dict(),
                }
        path = os.path.join(out_dir, f'checkpoint_{t}.pth')
        th.save(checkpoint, path)
        print(f'Checkpoint saved to {path}')
        return checkpoint
    def load_checkpoint(self, dir_path, chp_name=None):
        if chp_name in [None, 'last', 'end']:
            checkpoints = glob(os.path.join(dir_path, 'checkpoint_*.pth'))
            checkpoints.sort()
            chp_name = checkpoints[-1]
        elif chp_name in ['first', 'start']:
            chp_name = os.path.join(dir_path, 'checkpoint_0.pth')
        checkpoint = th.load(chp_name)
        t_load = float(os.path.basename(chp_name)[:-4].split('_')[1])
        self.t_load = t_load

        self.model.load_state_dict(checkpoint['model_sd'])
        self.optimizer.load_state_dict(checkpoint['optim_sd'])
        return checkpoint, t_load

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

    dir_path = get_timestamped_dir(load=True, base_dir='bars')
    solver = load_solver(dir_path, loader)
    #solver.load_checkpoint(dir_path)

    tspan = np.arange(1e3)

    soln = solver.solve(tspan, out_N=1e2)
    solver.save_soln(soln, 'bars')
    X = soln['x']
    R = soln['r']
    A = soln['A']

    show_img_XRA(X, R, A, img_shape=(H, W))
