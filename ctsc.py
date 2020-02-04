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
import matplotlib.pylab as plt

FILE_DIR = os.path.abspath(os.path.dirname(__file__))

def get_dt(tspan):
    dt = np.zeros_like(tspan)
    dt[1:] = tspan[1:] - tspan[:-1]
    return dt
def get_time_interval(tspan, N=None, dt=None):
    if not (N or dt):
        return None

    tspan -= min(tspan)
    if N is not None:
        dt = tspan[-1]/N

    interval = np.zeros_like(tspan, dtype=bool)

    for n in np.unique(tspan//dt):
        idx = np.argmin(np.abs(tspan - n*dt))
        interval[idx] = True
    return interval
def get_timestamped_dir(load=False, base_dir=None, dir_name=None):
    if dir_name is not None:
        dir_name = os.path.join(FILE_DIR, 'results', base_dir, dir_name)
        if not load:
            os.makedirs(dir_name)
        return dir_name

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
def load_solver(dir_path):
    with open(os.path.join(dir_path, 'params.yaml'), 'r') as f:
        params = yaml.safe_load(f)
    model_params = params['model_params']
    solver_params = params['solver_params']
    model = CTSCModel(**model_params)
    solver = CTSCSolver(model, **solver_params)
    return solver
def dsc_solver_param(n_A, n_s, eta_A, eta_s):
    tau_x = n_s
    t_max = n_A * n_s
    tau_s = 1 / eta_s
    tau_A = n_s / eta_A

    solver_params = dict(
            tau_A = tau_A,
            tau_u = tau_s,
            tau_x = tau_x,
            T_u = 0.,
            asynch=False,
            spike_coupling=True,
            t_max=t_max,
            )
    print('Converting to time constants...')
    for k, v in solver_params.items():
        print(f'{k} : {v}')
    return solver_params

    solver = cls(model, **solver_params)
    solver.t_max = t_max
    return solver

class CTSCModel(Module):
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
    def init_params(self):
        self.A = Parameter(th.Tensor(self.n_dim, self.n_dict))
        self.u = Parameter(th.Tensor(self.n_dict, self.n_batch))
        self.u0 = Parameter(th.tensor(self.u0))
        self.reset_params()
    def reset_params(self):
        self.A.data.normal_()
        self.A.data /= self.A.norm(dim=0)
        #self.A.data *= th.linspace(0.1, 0.5, self.n_dict)[None, :]
        self.u.data.normal_()
    @property
    def pi(self):
        return th.exp(-self.l1 * self.u0)
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
    def psnr(self, x):
        return 20 * th.log10(x.max()) - 10 * th.log10(self.recon_err(mean=True))
    def energy(self, x, A=None, u=None, return_recon=False):
            if not isinstance(x, th.Tensor):
                x = th.tensor(x)
            A0 = u0 = None
            if A is not None:
                A0 = self.A.data
                self.A.data = th.tensor(A)
            if u is not None:
                u0 = self.u.data
                self.u.data = th.tensor(u)
            recon = 0.5 * ((self.r - x)**2).sum()
            sparse = th.abs(self.u).sum()
            energy = recon/self.sigma**2 + self.l1 * sparse

            if A0 is not None:
                self.A.data = A0
            if u0 is not None:
                self.u.data = u0

            if return_recon:
                return energy, recon/self.sigma**2
            else:
                return energy
    def forward(self, x):
        return self.energy(x)

class CTSCSolver:
    @classmethod
    def load(cls, dir_path=None, base_dir=None, load_ckpt=False):
        dir_path = dir_path or get_timestamped_dir(load=True, base_dir=base_dir)
        with open(os.path.join(dir_path, 'params.yaml'), 'r') as f:
            params = yaml.safe_load(f)
        model_params = params['model_params']
        solver_params = params['solver_params']
        model = CTSCModel(**model_params)
        solver = CTSCSolver(model, **solver_params)
        if load_ckpt:
            if isinstance(load_ckpt, str):
                ckpt_name = load_ckpt
            else:
                ckpt_name = None
            solver.load_checkpoint(dir_path, ckpt_name)
        return solver
    @classmethod
    def get_dsc(cls, model, n_A, n_s, eta_A, eta_s, return_params=False):
        tau_x = n_s
        t_max = n_A * n_s
        tau_s = 1 / eta_s
        tau_A = n_s / eta_A

        solver_params = dict(
                tau_A = tau_A,
                tau_u = tau_s,
                tau_x = tau_x,
                T_u = 0.,
                asynch=False,
                spike_coupling=True,
                t_max=t_max,
                )
        print('Converting to time constants...')
        for k, v in solver_params.items():
            print(f'{k} : {v}')
        if return_params:
            return solver_params

        solver = cls(model, **solver_params)
        solver.t_max = t_max
        return solver
    def __init__(self, model, tau_A, tau_u, tau_x, mu_A=0., mu_u=0., T_A=0., T_u=1., asynch=False, spike_coupling=False, tau_A_correction=True, t_max=None, dt=1, tau_u0=None, mu_u0=0, T_u0=0):
        self.t_max = t_max
        self.dt = dt
        self.spike_coupling  = spike_coupling
        if spike_coupling:
            assert not asynch
        self.asynch = asynch

        # Record params
        self.params = {k:v for (k, v) in locals().items() if isinstance(v, (int, float, bool))}

        self.model = model
        self.n_batch = model.n_batch

        self.tau_x = tau_x

        if tau_A_correction:
            tau_A *= self.n_batch
        optim_params = [
                {'params':[model.A], 'tau':tau_A, 'mu':mu_A, 'T':T_A},
                {'params':[model.u], 'tau':tau_u, 'mu':mu_u, 'T':T_u},
                ]
        if tau_u0 is not None:
            optim_params.append(
                {'params':[model.u0], 'tau':tau_u0, 'mu':mu_u, 'T':T_u}
                    )
        self.optimizer = EulerMaruyama(optim_params)
        if tau_u0 is None:
            self.pg_A, self.pg_u = self.optimizer.param_groups
        else:
            self.pg_A, self.pg_u, _ = self.optimizer.param_groups
    def get_dir_path(self, base_dir, load=False, name=None, overwrite=False):
        if name is None:
            dir_path = get_timestamped_dir(load=load, base_dir=base_dir)
        else:
            dir_path = os.path.join(FILE_DIR, 'results', base_dir, name)
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
            elif (not overwrite) and (not load):
                raise Exception('Directory already exist')
        if not load:
            print(f'Experiment saved to {dir_path}')
            self.dir_path = dir_path
        return dir_path
    def load_batch_size(self, tspan):
        where_load = th.zeros(len(tspan), self.n_batch)
        if self.asynch:
            where_load.bernoulli_(1/self.tau_x)
        else:
            #load_idx = (tspan % self.tau_x == self.tau_x-1)
            load_idx = (tspan % self.tau_x == 0)

            where_load[np.argwhere(load_idx)] = 1
        return where_load
    def step_A_only(self, closure, dt):
        self.optimizer.zero_grad()
        self.pg_A['coupling'] = self.tau_x
        self.pg_u['coupling'] = 0
        self.optimizer.step(closure, dt=dt)
        self.pg_A['coupling'] = 0
        self.pg_u['coupling'] = 1
    def solve(self, loader, tmax=None, dt=None, normalize_A=True, soln_N=None, soln_T=None, soln_offset=0, save_N=None, save_T=None, callback_freq=None, callback_fn=None, out_mse=False, out_energy=False, norm_A_init=1):
        if self.model.A.is_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        if tmax is None:
            tmax = self.t_max
        if dt is None:
            dt = self.dt
        tspan = np.arange(int(tmax / dt))

        # Get time interval if number output/soln given
        if save_N is not None:
            assert save_T is None
            save_T = tmax // save_N
        if soln_N is not None:
            assert soln_T is None
            soln_T = max(tmax // soln_N, 1)

        # Get initial data x
        print('---')
        print(device)
        x = loader().to(device)

        # Define closure for solver
        def closure():
            energy = self.model(x)
            energy.backward()
            return energy
  
        # When to load new data
        load_n = self.load_batch_size(tspan)

        # Delta Ts in case dt != 1
        dtspan = get_dt(tspan)
        soln = defaultdict(list)

        # Supress A if DSC
        if self.spike_coupling:
            self.pg_A['coupling'] = 0

        if out_mse:
            mse_t = []
            mse_list = []

        # Iterate over tspan
        if out_energy:
            energy_arr = np.zeros_like(tspan)
        #self.model.A.data = self.model.A / self.model.A.norm(dim=0).mean() * norm_A_init
        if False:
            self.model.A.data = self.model.A / self.model.A.norm(dim=0)
            #_, n_dict = self.model.A.shape
            #self.model.A.data *= th.linspace(0.2, 0.8, n_dict)[None, :]
        for n, t in enumerate(tqdm(tspan)):
            # Optimizer step
            self.optimizer.zero_grad()
            if out_energy:
                energy_arr[n] = self.optimizer.step(closure, dt=dt)
            else:
                self.optimizer.step(closure, dt=dt)

            # Update again if DSC
            batch_size = int(load_n[n].sum())
            if self.spike_coupling and batch_size > 0:
                if n == 0:
                    continue
                self.step_A_only(closure, dt)

            # Normalize dictionary if needed
            if normalize_A:
                self.model.A.data = self.model.A / self.model.A.norm(dim=0)

            # Load new batch asynchronously if necessary
            if batch_size > 0:
                where_load = load_n[n].nonzero()
                if out_mse:
                    diff = (self.model.r[where_load] - x[where_load]).data.numpy()
                    mse_list.append((diff**2).mean())
                    mse_t.append(t)
                #TODO: remove this stupid switcharoo
                x_ = x.clone()
                x_.data[where_load] = loader(n_batch=batch_size).to(device).view(batch_size, 1, -1)
                x = x_

            # Call callback function
            if callback_freq and (t % callback_freq == 0):
                callback_fn(self.model)

            # Save Checkpoint
            if save_T is not None and (t % save_T == 0):
                self.save_checkpoint(t)

            # Save solution
            if soln_T is not None and ((t - soln_offset) % soln_T == 0):

                soln['r'].append(clone_numpy(self.model.r))
                soln['u'].append(clone_numpy(self.model.u))
                soln['s'].append(clone_numpy(self.model.s))
                soln['A'].append(clone_numpy(self.model.A))
                try:
                    soln['pi'].append(clone_numpy(self.model.pi))
                except:
                    pass
                print(f'pi : {self.model.pi:.2f}')
                print(f'norm : {self.model.A.norm(dim=0).mean():.2f}')
                try:
                    soln['x'].append(x.data.numpy())
                except:
                    soln['x'].append(x.cpu().data.numpy())

                soln['t'].append(t)
        if out_mse:
            soln['mse_t'] = np.array(mse_t)
            soln['mse'] = np.array(mse_list)
        if out_energy:
            soln['energy_t'] = np.array(tspan)
            soln['energy'] = np.array(energy_arr)

        if soln:
            for k, v in soln.items():
                soln[k] = np.array(v)
            return soln
        else:
            return None
    def save_hyperparams(self, dir_path=None):
        if dir_path is None:
            dir_path = self.dir_path
        params = dict(
                model_params=self.model.params,
                solver_params=self.params,
                )
        with open(os.path.join(dir_path, 'params.yaml'), 'w') as f:
            yaml.dump(params, f)
    def save_soln(self, soln, dir_path=None):
        if dir_path is None:
            dir_path = self.dir_path
        t_last = self.t_max
        #dir_path = get_timestamped_dir(base_dir=base_dir)
        self.save_hyperparams(dir_path)
        #self.save_checkpoint(t_last, dir_path=dir_path)
        soln_h5 = h5py.File(os.path.join(dir_path, 'soln.h5'), 'w')
        for k, v in soln.items():
            soln_h5.create_dataset(k, data=v)
    def save_checkpoint(self, t, dir_path=None):
        if dir_path is None:
            dir_path = self.dir_path
        checkpoint = {
                't': t,
                'model_sd': self.model.state_dict(),
                'optim_sd': self.optimizer.state_dict(),
                }
        path = os.path.join(dir_path, f'checkpoint_{t:.0f}.pth')
        th.save(checkpoint, path)
        print(f'Checkpoint saved to {path}')
        return checkpoint
    def load_checkpoint(self, dir_path, load_optim=False, chp_name=None):
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
        if load_optim:
            self.optimizer.load_state_dict(checkpoint['optim_sd'])
        return checkpoint, t_load
def clone_numpy(tensor):
    if tensor.is_cuda:
        tensor = tensor.to('cpu')
    return tensor.clone().data.numpy()

if __name__ == '__main__':
    from loaders import BarsLoader, VanHaterenSampler
    from visualization import show_img_XRA

    # Define loader
    H = W = 8
    N_DIM = H * W
    N_BATCH = int(N_DIM // 2)
    OC = 1.
    N_DICT = int(OC * N_DIM)
    PI = 0.3
    loader = VanHaterenSampler(H, W, N_BATCH)

    # DSC params
    N_A = 25
    N_S = 100
    ETA_A = 0.1
    ETA_S = 0.1

    # Define model, solver
    model_params = dict(
            n_dict = H * W,
            n_dim = H * W,
            n_batch = N_BATCH,
            positive=True,
            pi = 1,
            )
    solver_params = dsc_solver_param(n_A=N_A, n_s=N_S, eta_A=ETA_A, eta_s=ETA_S)

    model = CTSCModel(**model_params)
    solver = CTSCSolver(model, **solver_params)

    try:
        solver.get_dir_path('vh_test', name='trained_dict', overwrite=False)
        soln = solver.solve(loader, soln_N=1e4, save_N=1)
        solver.save_soln(soln)
    except:
        dir_path = os.path.join(FILE_DIR, 'results', 'vh_test', 'trained_dict')
        solver_params['tau_A'] = 1e9
        solver = CTSCSolver(model, **solver_params)
        solver.load_checkpoint(dir_path=dir_path)
        solver.get_dir_path('vh_test', name='infer', overwrite=True)
        soln = solver.solve(loader, out_N=1e3, save_N=1)


    X = soln['x'][:]
    R = soln['r'][:]
    A = soln['A'][:]

    show_img_XRA(X, R, A, n_frames=1e2, img_shape=(H, W))
