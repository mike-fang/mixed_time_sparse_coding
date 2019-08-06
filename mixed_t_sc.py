import numpy as np
import matplotlib.pylab as plt
from matplotlib import animation
from time import time
from loaders import Loader, Solutions_H5, Solutions
import h5py
import os.path
from time import time
from tqdm import tqdm
import pickle
import shutil

FILE_DIR = os.path.abspath(os.path.dirname(__file__))

class Energy_L0:
    def __init__(self, sigma, l0, l1, positive=False):
        self.sigma = sigma
        self.l0 = l0
        self.l1 = l1
        self.positive = positive
        self.s0 = -np.log(1 - l0) / l1
    def ds(self, s, x, A):
        sign_s = np.sign(s)
        where_active = (np.abs(s) >= self.s0)
        u = (s - self.s0*(sign_s)) * where_active
        dE_recon = (A.T @ (A @ u - x) / self.sigma**2) * where_active
        if self.positive:
            dE_sparse = self.l1 * ( (s > 0) - 10*(s < 0) )
        else:
            dE_sparse = self.l1 * sign_s
        return dE_recon + dE_sparse
    def dA(self, s, x, A):
        u = (s - self.s0*(np.sign(s))) * (np.abs(s) > self.s0)
        return (A @ u - x)[:, None] @ u[None, :] / self.sigma**2
    def __repr__(self):
        desc = f'''Energy_L0
l0 : {self.l0}
l1 : {self.l1}
s0 : {self.s0:.4f}
positive : {self.positive}
        '''
        return desc

def save_results(model, soln, loader, dir_name=None, overwrite=False):
    if dir_name in [None, 'tmp']:
        time_stamp = f'{time():.0f}'
        dir_name = os.path.join(FILE_DIR, 'results', 'tmp', time_stamp)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    else:
        if overwrite:
            print(f'Overwriting {dir_name}')
            shutil.rmtree(dir_name)
            os.mkdir(dir_name)
        else:
            raise Exception(f'Directory {dir_name} already exists')
    model_name = os.path.join(dir_name, 'model.pkl')
    soln_name = os.path.join(dir_name, 'soln.h5')
    param_name = os.path.join(dir_name, 'params.txt')
    solution = Solutions_H5(f_name=soln_name, solns=soln, im_shape=loader.im_shape)

    #solution.save(soln_name, overwrite=True)
    with open(model_name, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    with open(param_name, 'w') as f:
        f.write(model.get_desc(loader))
    return solution

class MixedTimeSC:
    @staticmethod
    def update_param(p, dEdx, dt, tau, mu, T=1, dW=None):
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
    def __init__(self, n_dim, n_sparse, energy, im_shape=None, **params):
        self.n_dim = n_dim
        self.n_sparse = n_sparse
        self.energy = energy
        self.params = params
        for k, v in params.items():
            setattr(self, k, v)
    def init_params(self, n_batch):
        A = np.random.normal(0, 0.4, size=(self.n_dim, self.n_sparse))
        s = np.zeros((n_batch, self.n_sparse))

        p_s = np.zeros((n_batch, self.n_sparse))
        p_A = np.zeros((self.n_dim, self.n_sparse))
        return s, A, p_s, p_A
    def train(self, loader, tspan, out_t=None, init_sA=None, dir_out='tmp'):
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

        # Precalculating Wiener process
        dW_s = np.random.normal(loc=0, scale= (2 * dT[:, None, None])**0.5, size=(t_steps - 1, loader.n_batch, self.n_sparse)) 

        # Init params and solns
        if init_sA is None:
            S, A, p_S, p_A = self.init_params(loader.n_batch)
        else:
            S, A, p_S, p_A = init_sA
        X = loader.get_batch()
        s_soln = np.zeros((n_out, loader.n_batch, self.n_sparse))
        x_soln = np.zeros((n_out, loader.n_batch, self.n_dim))
        A_soln = np.zeros((n_out, self.n_dim, self.n_sparse))
        s_soln[0] = S
        A_soln[0] = A
        x_soln[0] = X

        # Load up params
        mu_s = self.params['mu_s']
        tau_s = self.params['tau_s']
        mu_A = self.params['mu_A']
        tau_A = self.params['tau_A']

        # Iterate over time steps
        out_idx = 1
        for i, t in enumerate(tqdm(tspan[1:])):
            dt = dT[i]
            # Get new batch 
            if new_batch[i]:
                X = loader.get_batch()

            #Iterate over batch
            dA = 0
            dp_A = 0
            for j in range(loader.n_batch):
                # Calculate Gradient
                dEds = self.energy.ds(S[j], X[j], A)
                dEdA = self.energy.dA(S[j], X[j], A)

                # Claculate gradient
                dS, dp_S = self.update_param(p_S[j], dEds, dt, tau_s, mu_s, dW=dW_s[i, j])
                dAj, dp_Aj = self.update_param(p_A, dEdA, dt, tau_A, mu_A)

                # Update variables
                S[j] += dS
                p_S[j] += dp_S
                dA += dAj
                dp_A += dp_Aj

            # Update changes in A all at once, maybe sequentially with each s might be better
            A += dA / loader.n_batch
            p_A += dp_A / loader.n_batch

            # Record values
            if t in out_t:
                s_soln[out_idx] = S
                A_soln[out_idx] = A
                x_soln[out_idx] = X
                out_idx += 1

        sign_s = np.sign(s_soln)
        where_active = (np.abs(s_soln) >= self.energy.s0)
        u_soln = (s_soln - self.energy.s0*(sign_s)) * where_active

        solns = {}
        solns['X'] = x_soln
        #solns['S'] = s_soln
        solns['A'] = A_soln
        solns['T'] = out_t
        solns['S'] = u_soln

        #self.desc = self.get_desc(loader)
        return solns
    def show_evolution(self, soln, n_frames=100, overlap=3, f_out=None):
        fig, axes = plt.subplots(ncols=2, figsize=(14, 6))
        ax = axes[0]

        # Create scatter plots for s and x
        sx, sy = [], []
        xx, xy = [], []
        scat_s = ax.scatter(sx, sy, s=5, c='b', label=rf'$A \mathbf {{s}}$ : Reconstruction, $\tau_s = {self.tau_s} \tau$')
        scat_x = ax.scatter(xx, xy, s=50, c='r', label=rf'$\mathbf {{x}}$ : Data, $\tau_x = {self.tau_x}$')

        # Create arrows for tracking A
        A_arrow_0, = ax.plot([], [], c='g', label=rf'$A$ : Dictionary, $\tau_A = {self.tau_A} \tau$')
        A_arrows = [A_arrow_0]
        for A in range(self.n_sparse - 1):
            A_arrow, = ax.plot([], [], c='g')
            A_arrows.append(A_arrow)

        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect(1)
        fig.legend(loc='lower right')

        if False:
            # Create bar plot for sparse elements
            s_n = np.arange(len(soln['S'][0]))
            s_h = soln['S'][0]
            s_bar = axes[1].bar(s_n, s_h, fc='k')
            axes[1].set_ylim(-10, 10)
            axes[1].set_xticks(np.arange(self.n_sparse))


        idx_stride = int(len(soln['S'][:]) // n_frames)
        def animate(nf):
            idx0 = max(0, (nf - overlap + 1) * idx_stride)
            idx1 = (nf + 1) * idx_stride 

            T = soln['T'][idx0:idx1]
            y = soln['S'][idx0:idx1].reshape((-1, self.n_sparse))
            #y = y - np.sign(y)

            ti = T[0]
            tf = T[-1] 

            A = soln['A'][idx1]
        
            #u = (y - self.s0*(np.sign(y))) * (np.abs(y) > self.s0)
            u = y
            scat_s.set_offsets(u @ A.T)
            scat_s.set_array(np.linspace(0, 1, len(T)))
            scat_s.cmap = plt.cm.get_cmap('Blues')

            x = soln['X'][idx0:idx1].reshape((-1, self.n_dim))
            scat_x.set_offsets(x)

            for n in range(self.n_sparse):
                a = A[:, n] * 2
                A_arrows[n].set_xdata([0, a[0]])
                A_arrows[n].set_ydata([0, a[1]])

            fig.suptitle(rf'Time: ${ti:.2f} \tau - {tf:.2f} \tau$')

            if False:
                for i, b in enumerate(s_bar):
                    b.set_height(u[0, i])

        anim = animation.FuncAnimation(fig, animate, frames=n_frames-1, interval=100, repeat=True)
        if f_out is not None:
            anim.save(f_out)
        plt.show()
    def get_desc(self, loader):
        desc = "Mixed Time Sparse Coding Model\n"
        desc += f"n_dim: {self.n_dim}\n\
n_sparse: {self.n_sparse}\n\n"
        desc += f"-- ENERGY --\n {self.energy}\n"
        desc += f"-- LOADER --\n {loader}\n"
        desc += "-- PARAMS --\n"
        for k, v in self.params.items():
            desc += f'{k}: {v}\n'
        return desc

if __name__ == '__main__':
    # Hyper-params
    n_sparse = 3
    n_dim = 2
    params = {
            'tau_s': 1e2,
            'tau_x': 1e3,
            'tau_A': 1e4,
            'mu_s': .0,
            'mu_A': .05,
            }

    # Define energy
    l1 = .3
    l0 = .5
    sigma = .5
    energy = Energy_L0(sigma, l0, l1, positive=True)
    print(energy)

    # Build dataset
    theta = (np.linspace(0, 2*np.pi, n_sparse, endpoint=False))
    cos = np.cos(theta)
    sin = np.sin(theta)
    X = 10 * np.hstack((cos[:,None], sin[:,None]))
    n_batch = 3
    loader = Loader(X, n_batch)

    # Time range
    T_RANGE = 1e4
    T_STEPS = int(T_RANGE)
    tspan = np.linspace(0, T_RANGE, T_STEPS, endpoint=False)

    mt_sc = MixedTimeSC(n_dim, n_sparse, energy, **params)

    t0 = time()
    solns_dict = mt_sc.train(loader, tspan)
    solns = Solutions(solns_dict)

    #solns_db = save_soln(solns)
    mt_sc.show_evolution(solns_dict, n_frames=100)
