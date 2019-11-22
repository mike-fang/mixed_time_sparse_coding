from time import time
from collections import defaultdict
from euler_maruyama import EulerMaruyama
import torch as th
from torch.nn import Parameter
from torch.nn import Module
import torch.nn.functional as F
import numpy as np
from loaders import StarLoader, Loader, StarLoader_
from tqdm import tqdm
import matplotlib.pylab as plt
from matplotlib import animation
from visualization import show_2d_evo
import json
import os.path
from glob import glob
from solution_saver import Solutions
import sys
import traceback


FILE_DIR = os.path.abspath(os.path.dirname(__file__))

class MTSCModel(Module):
    # // Init //
    def __init__(self, n_dim, n_dict, n_batch, positive=False):
        super().__init__()
        self.n_dim = n_dim
        self.n_dict = n_dict
        self.n_batch = n_batch
        self.positive = positive

        self.init_params()
    def init_params(self, init=[None]):
        self.A = Parameter(th.Tensor(self.n_dim, self.n_dict))
        self.tau = Parameter(th.Tensor(1))
        self.l1 = Parameter(th.Tensor(1))
        self.rho = Parameter(th.Tensor(1))

        self.s_data = Parameter(th.Tensor(self.n_dict, self.n_batch))
        self.x_model = Parameter(th.Tensor(self.n_batch, self.n_dim))
        self.s_model = Parameter(th.Tensor(self.n_dict, self.n_batch))

        self.reset_params(init=init)
    def reset_params(self, init=[None]):
        self.A.data.normal_()
        self.s_data.data.normal_()
        self.s_model.data.normal_()
        self.x_model.data.normal_()

        self.tau.data = th.tensor(1.)
        self.l1.data = th.tensor(1.)
        self.rho.data = th.tensor(0.6)
        if init != [None]:
            for n in init:
                init[n] = th.tensor(init[n], dtype=th.float)
        for n, p in self.named_parameters():
            if n in init:
                p.data = init[n]
        if 'pi' in init:
            self.pi = init['pi']
        if 'sigma' in init:
            self.tau.data = init['sigma']**(-1)
    # // Param conversions // 
    @property
    def pi(self):
        return th.sigmoid(self.rho.data)
    @pi.setter
    def pi(self, pi):
        pi = th.tensor(pi, dtype=th.float)
        self.rho.data = th.log(pi / (1 - pi))
    @property
    def s0(self):
        return F.softplus(-self.rho) / self.l1
    # // Energy definition //
    def get_recon(self, s):
        if self.positive:
            u = F.relu(s - self.s0) + F.relu(-s - self.s0)
        else:
            u = F.relu(s - self.s0) - F.relu(-s - self.s0)
        return (self.A @ u).T
    def sc_energy(self, s, x):
        r = self.get_recon(s)
        recon_loss = 0.5 * ((x - r)**2).sum()
        sparse_loss = th.abs(s).sum() 
        return (self.tau * recon_loss + self.l1 * sparse_loss) 
    def energy(self, x_data):
        return self.sc_energy(self.s_data, x_data)# - 0*self.sc_energy(self.s_model, self.x_model)
    def forward(self, x):
        return self.energy(x)

class MTSCModel_NoModel(Module):
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
        self.tau = Parameter(th.Tensor(1))
        self.l1 = Parameter(th.Tensor(1))
        self.rho = Parameter(th.Tensor(1))

        # nu = log(pi / (1-pi)) log odds ratio
        self.reset_params(init=init)
    def reset_params(self, init=[None]):
        self.A.data.normal_()
        self.s.data.normal_()
        #self.s.data *= 0
        self.tau.data = th.tensor(1.)
        self.l1.data = th.tensor(1.)
        self.rho.data = th.tensor(0.6)
        if init != [None]:
            for n in init:
                init[n] = th.tensor(init[n], dtype=th.float)
        for n, p in self.named_parameters():
            if n in init:
                p.data = init[n]
        if 'pi' in init:
            self.pi = init['pi']
        if 'sigma' in init:
            self.tau.data = init['sigma']**(-1)
    @property
    def pi(self):
        return th.sigmoid(self.rho.data)
    @pi.setter
    def pi(self, pi):
        pi = th.tensor(pi, dtype=th.float)
        self.rho.data = th.log(pi / (1 - pi))
    @property
    def s0(self):
        return F.softplus(-self.rho) / self.l1
    @property
    def u(self):
        if self.positive:
            u = F.relu(self.s - self.s0) + F.relu(-self.s - self.s0)
        else:
            u = F.relu(self.s - self.s0) - F.relu(-self.s - self.s0)
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
        return self.tau * self.recon_error(x) + self.l1 * self.sparse_loss()
    def forward(self, x):
        return self.energy(x)

# Misc Fns
def train_dsc(tmax, tau_x, model, solver, loader, t_start=0, n_soln=None, out_dir=None, t_save=None, normalize_A=True, alpha=None):
    tmax = int(tmax)
    tau_x = int(tau_x)
    t_start = int(t_start)
    if (n_soln is None) or (n_soln > tmax):
        n_soln = tmax
    when_out = np.linspace(t_start, tmax+1, num=n_soln+1, dtype=int)[:-1]
    
    # If t_save not specified, save only begining and end
    if (out_dir is not None) and (t_save is None):
        t_save = tmax

    # Define soln
    soln = defaultdict(list)

    # train model
    #x = loader()
    def closure():
        energy = model(x)
        energy.backward()
        return energy

    for p in solver.param_groups:
        try:
            if model.A in p['params']:
                A_group = p
        except:
            pass
        try:
            if model.s_data in p['params']:
                s_group = p
        except:
            pass
    for t in tqdm(range(t_start, tmax + 1)):
        if t % int(tau_x) == 0:
            x = loader()
            A_group['coupling'] = 1
            s_group['coupling'] = 0
        else:
            A_group['coupling'] = 0
            s_group['coupling'] = 1
        solver.zero_grad()
        energy = float(solver.step(closure))
        if normalize_A:
            model.A.data = model.A / model.A.norm(dim=0)
            #A = model.A.data.numpy()
            #_, S, _ = np.linalg.svd(A)
        if t in when_out:
            for n, p in model.named_parameters():
                soln[n].append(p.clone().data.cpu().numpy())
            soln['r_model'].append(model.get_recon(model.s_model).data.numpy())
            soln['r_data'].append(model.get_recon(model.s_data).data.numpy())
            soln['x_data'].append(x.data.numpy())
            soln['t'].append(t)
            soln['energy'].append(energy)
        if (t_save is not None) and (t % t_save == 0):
            save_checkpoint(model, solver, t, out_dir)

    for k, v in soln.items():
        soln[k] = np.array(v)
    return soln
def get_timestamped_dir(load=False, base_dir=None):
    if base_dir is None:
        base_dir = 'tmp'
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
def get_model_solver(model_params=None, solver_params=None, path=None, noise_cache=0, model_class=MTSCModel):
    if path is not None:
        with open(os.path.join(path), 'r') as f:
            hp = json.load(f)
        model_params = hp['model_params']
        solver_params = hp['solver_params']
    model = model_class(**model_params)
    param_groups = get_param_groups(solver_params, model)
    solver = EulerMaruyama(param_groups, noise_cache=noise_cache)
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
def train_mtsc(tmax, tau_x, model, solver, loader, t_start=0, n_soln=None, out_dir=None, t_save=None, normalize_A=True, coupling='const', coupling_scale='tmax', alpha=None, rand_tau_x=False):
    if callable(coupling):
        # Already given a function
        pass
    elif coupling in ['const', None]:
        alpha = None
    elif coupling == 'exp':
        def coupling(t_):
            return (1 - np.exp(-t_ / alpha))
    elif coupling == 'exp_decay':
        def coupling(t_):
            return np.exp(-t_ / alpha)
    elif coupling == 'exp_':
        def coupling(t_):
            return np.exp(t_/alpha) / (alpha * (np.exp(1 / alpha) - 1))
    elif coupling == 'step':
        def coupling(t_):
            return (t_ > alpha) * 1.
    else:
        raise Exception('Invalid coupling type')

    if coupling_scale == 'tmax':
        coupling_scale = tmax
    elif coupling_scale in ['x', 'tau_x']:
        coupling_scale = tau_x

    tmax = int(tmax)
    tau_x = int(tau_x)
    t_start = int(t_start)
    if (n_soln is None) or (n_soln > tmax):
        n_soln = tmax
    when_out = np.linspace(t_start, tmax+1, num=n_soln+1, dtype=int)[:-1]
    
    # If t_save not specified, save only begining and end
    if (out_dir is not None) and (t_save is None):
        t_save = tmax

    # Define soln
    soln = defaultdict(list)

    # train model
    x = loader()
    def closure():
        energy = model(x)
        energy.backward()
        return energy

    for p in solver.param_groups:
        try:
            if model.A in p['params']:
                A_group = p
        except:
            pass
    x0 = x
    if rand_tau_x:
        load_x = th.LongTensor(tmax, loader.n_batch).bernoulli_(1/tau_x)
    for t in tqdm(range(t_start, tmax)):
        if rand_tau_x:
            load_idx = load_x[t]
            if load_idx.sum() > 0:
                x_ = x.clone()
                x_[load_idx.nonzero()] = loader()[load_idx.nonzero()]
                x = x_
        if t % int(tau_x) == 0:
            if not rand_tau_x:
                x = loader()
            overflow = ( th.any(th.isinf(model.s_data)) or th.any(th.isnan(model.s_data)) )
            assert not overflow
        if alpha is not None:
            t_ = (t % coupling_scale) / coupling_scale * alpha
            A_group['coupling'] = coupling(t_)

        solver.zero_grad()
        energy = float(solver.step(closure))
        if normalize_A:
            model.A.data = model.A / model.A.norm(dim=0)
            #A = model.A.data.numpy()
            #_, S, _ = np.linalg.svd(A)
        if t in when_out:
            for n, p in model.named_parameters():
                soln[n].append(p.clone().data.cpu().numpy())
            soln['r_model'].append(model.get_recon(model.s_model).data.numpy())
            soln['r_data'].append(model.get_recon(model.s_data).data.numpy())
            soln['x_data'].append(x.data.numpy())
            soln['t'].append(t)
            soln['energy'].append(energy)
        if (t_save is not None) and (t % t_save == 0):
            save_checkpoint(model, solver, t, out_dir)

    for k, v in soln.items():
        soln[k] = np.array(v)
    return soln

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
    def __init__(self, model_params, solver_params, base_dir=None, im_shape=None, noise_cache=0, trainer=train_mtsc, alpha=None, coupling='const'):
        self.set_model_solver(model_params, solver_params)
        self.base_dir = base_dir
        self.im_shape = im_shape
        self.alpha = alpha
        self.trainer = trainer
        self.coupling = coupling

        if noise_cache > 0:
            self.solver.reset_noise(noise_cache=noise_cache)
    @property
    def hyperparams(self):
        try:
            tau_x = self.tau_x
        except:
            tau_x = 'none'
        try:
            tmax = self.tmax
        except:
            tmax = 'none'
        return dict(tmax=tmax, tau_x = tau_x, model_params=self.model_params, solver_params=self.solver_params)
    def save_hyperparams(self):
        with open(os.path.join(self.dir_path, 'hyperparams.json'), 'w') as f:
            json.dump(self.hyperparams, f)
    def set_model_solver(self, model_params, solver_params):
        self.model_params = model_params
        self.solver_params = solver_params
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
    def start_new_soln(self, tmax, tstart=0, n_soln=None, t_save=None, soln_name=None, rand_tau_x=False):
        if soln_name is None:
            self.dir_path = get_timestamped_dir(base_dir=self.base_dir)
            print(f'Saving experiment to directory {self.dir_path}')
            name = os.path.join(self.dir_path, 'soln.h5')
            self.tmax = tmax
            self.save_hyperparams()
            if os.path.isfile(name):
                for n in range(100):
                    soln_name = os.path.join(self.dir_path, f'soln_{n}.h5')
                    if not os.path.isfile(soln_name):
                        break
            else:
                soln_name = name
        soln_dict = self.trainer(tmax=tmax, t_start=tstart, loader=self.loader, tau_x=self.tau_x, model=self.model, solver=self.solver, n_soln=n_soln, out_dir=self.dir_path, t_save=t_save, alpha=self.alpha, coupling=self.coupling, rand_tau_x=rand_tau_x)
        soln = Solutions(soln_dict, f_name=soln_name, im_shape=self.im_shape)
        return soln
    def load_soln(self, f_name=None):
        if f_name is None:
            f_name = os.path.join(self.dir_path, 'soln.h5')
        soln = Solutions.load(f_name)
        return soln
    def load(self, path=None):
        if path is None:
            dir_path = get_timestamped_dir(load=True, base_dir=self.base_dir)
        else:
            dir_path = os.path.join('./results', self.base_dir, path)

        with open(os.path.join(dir_path, 'hyperparams.json'), 'r') as f:
            hp = json.load(f)
        model_params = hp['model_params']
        solver_params = hp['solver_params']
        self.set_model_solver(model_params, solver_params)
        self.dir_path = dir_path
        return Solutions.load(os.path.join(self.dir_path, 'soln.h5'))
        #return cls(model_params, solver_params, dir_path)

class DSCSolver(MTSCSolver):
    def __init__(self, model_params, solver_params, **kwargs):
        super().__init__(model_params, solver_params, **kwargs, trainer=train_dsc)
        print(self.trainer)

if __name__ == '__main__':
    PI = .5
    L1 = 1
    SIGMA = 0
    N_BATCH = 15
    new = True
    tmax = int(1e4)
    tau_x = int(1e3)
    tau_s = int(1e2)
    tau_A = 1e4
    loader = StarLoader(n_basis=3, n_batch=N_BATCH, sigma=1, pi=PI, l1=.2, coeff='exp')
    print(loader())
    model_params = dict(
            n_dict = 3,
            n_dim = 2,
            n_batch = N_BATCH,
            positive = True
            )
    solver_params = [
            dict(params = ['s_data'], tau=tau_s, T=1),
            #dict(params = ['s_model'], tau=-tau_s/5, T=0),
            #dict(params = ['x_model'], tau=-tau_x/5, T=1),
            dict(params = ['A'], tau=tau_A*N_BATCH),
            ]
    init = dict(
            pi = PI,
            l1 = L1,
            sigma = 1,
            )
    try:
        if new:
            assert False
        soln = Solutions.load()
    except:
        mtsc_solver = MTSCSolver(model_params, solver_params, coupling='exp', alpha=.2)
        mtsc_solver.model.reset_params(init=init)
        mtsc_solver.set_loader(loader, tau_x)
        soln = mtsc_solver.start_new_soln(tmax=tmax, n_soln=1000)

    show_2d_evo(soln)
