import numpy as np
from loaders import BarsLoader
from tqdm import tqdm
import matplotlib.pylab as plt
from ctsc import CTSCSolver
from collections import defaultdict


class LCA:
    def __init__(self, n_dim, n_dict, n_batch, u0=1, positive=True):
        self.n_dim = n_dim
        self.n_dict = n_dict
        self.n_batch = n_batch
        self.u0 = u0
        self.positive = positive
        self.reset_params()
    def reset_params(self):
        self.A = np.random.normal(0, 1, size=(self.n_dim, self.n_dict))
        self.u = np.random.normal(0, 1, size=(self.n_dict, self.n_batch))
    @property
    def s(self):
        where_thresh = np.abs(self.u) <= self.u0
        s = np.zeros_like(self.u)
        s[~where_thresh] = self.u[~where_thresh] - self.u0 * np.sign(self.u[~where_thresh])
        if self.positive:
            s = np.abs(s)
        return s
    @property
    def r(self):
        return self.A @ self.s
    def step_u(self, x, eta):
        sign = np.sign(self.u) if self.positive else 1
        grad = self.A.T @ (self.r - x.T) * sign + (self.u - self.s)
        self.u -= eta * grad
    def step_A(self, x, eta, normalize=True):
        grad = (self.r - x.T) @ self.s.T
        self.A -= eta * grad / self.n_batch
        if normalize:
            self.A /= np.linalg.norm(self.A, axis=0)
    def rmse(self, x):
        err = ((self.A @ self.s - x)**2).mean()**0.5
        return err

class LCASolver(CTSCSolver):
    def __init__(self, model, n_A, n_s, eta_A, eta_s):
        n_A = int(n_A)
        n_s = int(n_s)
        self.tau_x = n_s
        self.t_max = n_A * n_s
        self.eta_A = eta_A
        self.eta_s = eta_s

        # Record params
        self.params = {k:v for (k, v) in locals().items() if isinstance(v, (int, float, bool))}

        self.model = model
        self.n_batch = model.n_batch
    def solve(self, loader, tmax=None, out_N=None, out_T=None, save_N=None, save_T=None, callback_freq=None, callback_fn=None):
        if tmax is None:
            tmax = self.t_max
        tspan = np.arange(tmax)

        # Get time interval if number output/soln given
        if save_N is not None:
            assert save_T is None
            save_T = tmax // save_N
        if out_N is not None:
            assert out_T is None
            out_T = max(tmax // out_N, 1)

        # Get initial data x
        x = loader()


        soln = defaultdict(list)

        # Iterate over tspan
        for t in tqdm(tspan):
            self.model.step_u(x, self.eta_s)
            if t % self.tau_x == 0:
                self.model.step_A(x, self.eta_A)
                x = loader()


            # Call callback function
            if callback_freq and (t % callback_freq == 0):
                callback_fn(self.model)

            # Save Checkpoint
            if save_T is not None and (t % save_T == 0):
                self.save_checkpoint(t)

            # Save solution
            if out_T is not None and (t % out_T == 0):
                soln['r'].append(self.model.r)
                soln['u'].append(self.model.u)
                soln['s'].append(self.model.s)
                soln['A'].append(self.model.A)
                soln['x'].append(x)
                soln['t'].append(t)

        if soln:
            for k, v in soln.items():
                soln[k] = np.array(v)
            return soln
        else:
            return None

if __name__ == '__main__':


    H = W = 8
    N_DIM = H * W
    N_BATCH = 80
    N_DICT = H + W
    PI = 0.3
    loader = BarsLoader(H, W, N_BATCH, p=PI, numpy=True)

    N_A = 2500
    N_S = 20
    eta_A = .1
    eta_S = 0.1

    U0 = 0.20

    lca = LCA(n_dim=N_DIM, n_dict=N_DICT, n_batch=N_BATCH, u0=U0, positive=True)
    lca_solver = LCASolver(lca, N_A, N_S, eta_A, eta_S)
    soln = lca_solver.solve(loader, out_N=1e3)
    
    A = soln['A'][-1]
    fig, axes = plt.subplots(nrows=4, ncols=4)
    axes = [a for row in axes for a in row]
    for n, ax in enumerate(axes):
        ax.imshow(A[:, n].reshape(H, W))
    plt.show()
