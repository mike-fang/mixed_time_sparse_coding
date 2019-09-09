from time import time
from collections import defaultdict
from euler_maruyama import EulerMaruyama
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
import json
import os.path
from glob import glob
from solution_saver import Solutions

FILE_DIR = os.path.abspath(os.path.dirname(__file__))

class SCModelL0(Module):
    def __init__(self, n_dim, n_dict, n_batch, positive=False):
        super().__init__()
        self.n_dim = n_dim
        self.n_dict = n_dict
        self.n_batch = n_batch
        self.positive = positive

        self.init_params()
    def init_params(self, init=[None]):
        self.A = Parameter(th.Tensor(self.n_dim, self.n_dict))
        self.s = Parameter(th.Tensor(self.n_dict, self.n_batch))
        self.rho = Parameter(th.Tensor(1))
        self.l1 = Parameter(th.Tensor(1))
        self.nu = Parameter(th.Tensor(1))

        # nu = log(pi / (1-pi)) log odds ratio
        self.reset_params(init=init)
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
    @property
    def u(self):
        if self.positive:
            beta = 1
            s = F.softplus(beta * self.s) 
        else:
            s = self.s
        if True:
            s0 = -self.l1 * th.log(th.sigmoid(-self.nu))
            u = F.relu(s - s0) - F.relu(-s - s0)
        else:
            u = self.s
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
    def forward(self, x):
        return self.energy(x)

def get_timestamped_dir(load=False, base_dir='tmp'):
    if load:
        tmp_files = glob(os.path.join(FILE_DIR, 'results', base_dir, '*'))
        tmp_files.sort()
        dir_name = tmp_files[-1]
    else:
        time_stamp = f'{time():.0f}'
        dir_name = os.path.join(FILE_DIR, 'results', base_dir, time_stamp)
        os.makedirs(dir_name)
    return dir_name
def get_param_groups(solver_params, model): 
    param_groups = []
    for sp in solver_params:
        pg = dict(sp)
        params = []
        for n, p in model.named_parameters():
            if n in sp['params']:
                params.append(p)
        pg['params'] = params
        param_groups.append(pg)
    return param_groups
def get_model_solver(model_params=None, solver_params=None, path=None):
    if path is not None:
        with open(os.path.join(path), 'r') as f:
            hp = json.load(f)
        model_params = hp['model_params']
        solver_params = hp['solver_params']
    model = SCModelL0(**model_params)
    param_groups = get_param_groups(solver_params, model)
    solver = EulerMaruyama(param_groups)
    return model, solver
def save_checkpoint(model, solver, t, out_dir, f_name=None):
    checkpoint = {
            't': t,
            'model_sd': model.state_dict(),
            'solver_sd': solver.state_dict(),
            }
    if f_name is None:
        f_name = f'checkpoint_{t}.pth'
    out_path = os.path.join(out_dir, f_name)
    th.save(checkpoint, out_path)
    print(f'Checkpoint saved to {out_path}')
    return checkpoint
def train_mtsc(tmax, tau_x, model, solver, loader, t_start=0, n_soln=None, out_dir=None, t_save=None):
    tmax = int(tmax)
    tau_x = int(tau_x)
    t_start = int(t_start)
    if n_soln is None:
        n_soln = tmax
    when_out = (tmax / n_soln * np.arange(n_soln)).astype(int)
    
    # If t_save not specified, save only begining and end
    if (out_dir is not None) and (t_save is None):
        t_save = tmax

    # Define soln
    soln = defaultdict(list)
    def save_soln():
        for n, p in model.named_parameters():
            soln[n].append(p.clone().data.cpu().numpy())
        soln['r'].append(model.r.data.numpy())
        soln['x'].append(x.data.numpy())
        soln['t'].append(t)

    # train model
    def closure():
        energy = model(x)
        energy.backward()
        return energy
    for t in tqdm(range(t_start, tmax + 1)):
        if t % int(tau_x) == 0:
            x = loader()
        solver.zero_grad()
        solver.step(closure)
        if t in when_out:
            save_soln()
        if (t_save is not None) and (t % t_save == 0):
            save_checkpoint(model, solver, t, out_dir)

    for k, v in soln.items():
        soln[k] = np.array(v)
    return soln

    def get_model_solver(model_dims, solver_params):
        model = SCModelL0(**model_dims)

class MTSCSolver:
    @classmethod
    def load(cls, dir_path=None):
        if dir_path is None:
            dir_path = get_timestamped_dir(load=True)
        elif not os.path.isdir(dir_path):
            dir_path = get_timestamped_dir(load=True, base_dir=dir_path)
        with open(os.path.join(dir_path, 'hyperparams.json'), 'r') as f:
            hp = json.load(f)
        model_params = hp['model_params']
        solver_params = hp['solver_params']
        return cls(model_params, solver_params, dir_path)
    def __init__(self, model_params, solver_params, dir_path=None, im_shape=None):
        self.model_params = model_params
        self.solver_params = solver_params
        if dir_path is None:
            dir_path = get_timestamped_dir()
        elif not os.path.isdir(dir_path):
            dir_path = get_timestamped_dir(base_dir=dir_path)
        print(f'Saving experiment to directory {dir_path}')
        self.dir_path = dir_path
        self.im_shape = im_shape

        self.save_hyperparams()
        self.set_model_solver()
    @property
    def hyperparams(self):
        return dict(model_params=self.model_params, solver_params=self.solver_params)
    def save_hyperparams(self):
        with open(os.path.join(self.dir_path, 'hyperparams.json'), 'w') as f:
            json.dump(self.hyperparams, f)
    def set_model_solver(self):
        self.model, self.solver = get_model_solver(self.model_params, self.solver_params)
    def set_loader(self, loader, tau_x):
        self.loader = loader
        self.tau_x = tau_x
    def load_checkpoint(self, chp_name=None, auto_load=True):
        if chp_name in [None, 'last', 'end']:
            checkpoints = glob(os.path.join(self.dir_path, 'checkpoint_*.pth'))
            checkpoints.sort()
            chp_name = checkpoints[-1]
        elif chp_name in ['first', 'start']:
            chp_name = os.path.join(self.dir_path, 'checkpoint_0.pth')
        checkpoint = th.load(chp_name)
        if auto_load:
            self.model.load_state_dict(checkpoint['model_sd'])
            self.solver.load_state_dict(checkpoint['solver_sd'])
        return checkpoint
    def start_new_soln(self, tmax, tstart=0, n_soln=None, t_save=None, soln_name=None):
        if soln_name is None:
            name = os.path.join(self.dir_path, 'soln.h5')
            if os.path.isfile(name):
                for n in range(100):
                    soln_name = os.path.join(self.dir_path, f'soln_{n}.h5')
                    if not os.path.isfile(soln_name):
                        break
            else:
                soln_name = name
        soln_dict = train_mtsc(tmax=tmax, t_start=tstart, loader=self.loader, tau_x=self.tau_x, model=self.model, solver=self.solver, n_soln=n_soln, out_dir=self.dir_path, t_save=t_save)
        soln = Solutions(soln_dict, f_name=soln_name, im_shape=self.im_shape)
        return soln
    def load_soln(self, f_name=None):
        if f_name is None:
            f_name = os.path.join(self.dir_path, 'soln.h5')
        soln = Solutions.load(f_name)
        return soln

if __name__ == '__main__':
    model_params = dict(
            n_dict = 3,
            n_dim = 2,
            n_batch = 3,
            positive = True
            )
    solver_params = [
            dict(params = ['s'], tau=1e2, T=1),
            dict(params = ['A'], tau=1e5),
            ]
    init = dict(
            nu = 0.5,
            )
    tmax = int(1e3)
    tau_x = int(5e2)
    loader = StarLoader(n_basis=3, n_batch=model_params['n_batch'])

    if False:
        mtsc_solver = MTSCSolver(model_params, solver_params)
        mtsc_solver.model.reset_params(init=init)
        mtsc_solver.set_loader(loader, tau_x)
        soln = mtsc_solver.start_new_soln(tmax=tmax, n_soln=10000)
    else:
        mtsc_solver = MTSCSolver.load()
        mtsc_solver.load_checkpoint(chp_name='start')
        mtsc_solver.set_loader(loader, tau_x)
        soln = mtsc_solver.start_new_soln(tmax=tmax, n_soln=10000)
    show_2d_evo(soln, n_frames=100)
