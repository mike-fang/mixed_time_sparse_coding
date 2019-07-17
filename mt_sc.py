import torch as th
from torch.nn.parameter import Parameter
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt
from matplotlib import animation
import os.path as osp
from time import time

def ito_euler(f, g, x0, tspan):
    """
    Solves SDE of the form:
        dx = f(x, t) * dt + g(x, t) * dW
    with Euler-Maruyama. 

    args:
        f: Function with args x, t
        g: Function with args x, t
        x0: Initialization of x
        tspan: The points in time to solve the SDE

    returns:
        y - The solution to the SDE as a pytorch tensor

    """
    # Convert to torch Tensor if not already
    if not isinstance(x0, th.Tensor):
        x0 = th.tensor(x0).float()
    if not isinstance(tspan, th.Tensor):
        tspan = th.tensor(tspan).float()

    # Get shape of x
    x_shape = list(x0.shape)
    if x_shape == []:
        n_dim = 1
    else:
        n_dim = x_shape[0]

    dt = tspan[1:] - tspan[:-1]
    dW = th.zeros((len(dt), n_dim))
    dW.normal_()
    dW *= dt[:, None]**0.5

    x = x0

    if n_dim == 1:
        y = th.zeros_like(tspan)
    else:
        y = th.zeros((len(tspan), n_dim))
    y[0] = x0

    for i, t in enumerate(tspan[1:]):
        a = th.tensor(f(x, t))
        b = th.tensor(g(x, t))
        if n_dim == 1:
            x = x + a * dt[i] + b * dW[i]
        else:
            x = x + a @ dt[i] + b @ dW[i]
        y[i] = x

    return y

class MixT_SC:
    def __init__(self, tau_s, tau_x, tau_A, sigma_n, sparsity, l0):
        self.tau_s = tau_s
        self.tau_x = tau_x
        self.tau_A = tau_A
        self.sigma_n = sigma_n
        self.sparsity = sparsity
        self.s0 = -np.log(1 - l0) / sparsity
        print(self.s0)
    def set_X(self, X):
        if not isinstance(X, th.Tensor):
            X = th.Tensor(X)

        N, D = X.shape
        self.n_data = N
        self.n_dim = D
        self.X = X
        self.shuffle_X()
    def shuffle_X(self):
        t_chunks = (self.tspan//self.tau_x).long()
        uniq_idx = th.unique(t_chunks)
        n_batches = np.ceil(len(th.unique(t_chunks))/self.n_data).astype(int)
        x_idx = th.zeros_like(t_chunks)
        for i in range(n_batches):
            batches = uniq_idx[i*self.n_data:(i+1)*self.n_data]
            rand_idx = th.randperm(self.n_data)
            for n, j in enumerate(batches):
                x_idx[t_chunks == j] = rand_idx[n]

        self.X_shuffled = self.X[x_idx]
    def init_s(self):
        self.s = Parameter(th.zeros(self.n_sparse))
        return self.s
    def init_A(self):
        self.A = Parameter(th.Tensor(self.n_dim, self.n_sparse))
        self.A.data.normal_()
        self.A.data *= .4
        #self.A = Parameter(th.eye(2))
        return self.A
    def get_x_idx(self, t):
        try:
            return ( (t//self.tau_x) % self.n_data ).long()
        except:
            return ( (t//self.tau_x) % self.n_data ).astype(int)
    def H_SS(self, A, s, x):
        s0 = self.s0
        u = (s - s0*(th.sign(s))) * (th.abs(s) > s0).float()
        return self.H_reconstr(A, u, x) + self.H_sparse_L1_pos(s)
        return self.H_reconstr(A, u, x) + self.H_sparse_L1(s)
    def H_L1(self, A, s, x):
        return self.H_reconstr(A, s, x) + self.H_sparse_L1(s)
    H_ = H_SS
    def H_reconstr(self, A, s, x):
        s0 = self.s0
        return 0.5 * th.norm(x - A @ s)**2 / self.sigma_n**2
    def H_well(self, s):
        H =  (s**2 - self.s0**2) * (th.abs(s) < self.s0).float()
        return H.sum()
    def H_sparse_L1_pos(self, s):
        return s[s > 0].sum() * self.sparsity - s[s < 0].sum() * self.sparsity * 10
    def H_sparse_L1(self, s):
        return th.norm(s, p=1) * self.sparsity
    def H_sparse_L2(self, s):
        return th.norm(s, p=2) * self.sparsity
    def train(self, X, tspan, n_sparse):
        t0 = time()
        self.n_sparse = n_sparse
        self.tspan = tspan
        self.t_steps = len(tspan)
        self.set_X(X)

        A = self.init_A()
        s = self.init_s()

        s_soln, A_soln = self.solve( n_sparse)
        print(f'Training took {(time() - t0)*1e3} ms')
        return s_soln, A_soln
    def zero_grad(self):
        for p in [self.s, self.A]:
            p.grad.detach_()
            p.grad.zero_()
    def solve(self, n_sparse):
        A = self.A
        s = self.s
        dt = self.tspan[1:] - self.tspan[:-1]
        dW = th.zeros((len(dt), n_sparse))
        dW.normal_()
        dW *= (2 *dt[:, None])**0.5

        if n_sparse == 1:
            #s_soln = th.zeros_like(tspan)
            s_soln = th.zeros(self.t_steps)
        else:
            s_soln = th.zeros((self.t_steps, self.n_sparse))
        A_soln = th.zeros((self.t_steps, self.n_dim, self.n_sparse))
        s_soln[0] = s.data
        A_soln[0] = A.data

        for i, t in enumerate(self.tspan[1:]):
            x = self.X_shuffled[i]
            #H_total = self.H_reconstr(A, s, x) + self.H_sparse(s)
            H_total = self.H_(A, s, x)
            H_total.backward()
            s.data -= s.grad * dt[i] / self.tau_s
            s.data += dW[i] / self.tau_s**0.5

            #coupling_A_x = 1 - np.exp(-15 * self.tau_x * (t % self.tau_x))
            coupling_A_x = t % self.tau_x > self.tau_x/5
            A.data -= coupling_A_x * A.grad * dt[i] / self.tau_A

            s_soln[i] = s.data
            A_soln[i] = A.data
            self.zero_grad()

        return s_soln, A_soln
    def save_evolution(self, s_soln, A_soln, tspan, n_frames=100, overlap=3, f_out=None):
        assert len(s_soln) == len(tspan) == len(A_soln)

        if isinstance(s_soln, th.Tensor):
            s_soln = s_soln.data.numpy()
        if isinstance(A_soln, th.Tensor):
            A_soln = A_soln.data.numpy()
        if isinstance(tspan, th.Tensor):
            tspan = tspan.data.numpy()

        X = self.X[self.get_x_idx(tspan)]

        fig, axes = plt.subplots(ncols=2, figsize=(14, 6))
        ax = axes[0]

        sx, sy = [], []
        xx, xy = [], []
        scat_s = ax.scatter(sx, sy, s=5, c='b', label=rf'$A \mathbf {{s}}$ : Reconstruction, $\tau_s = {tau_s} \tau$')
        scat_x = ax.scatter(xx, xy, s=50, c='r', label=rf'$\mathbf {{x}}$ : Data, $\tau_x = {tau_x}$')

        a1 = A_soln[0, :, 0] * 5
        a2 = A_soln[0, :, 1] * 5
        _, _, n_sparse = A_soln.shape
        #line_A_1, = ax.plot([], [], c='g', label=rf'$A$ : Dictionary, $\tau_A = {tau_A} \tau$')
        A_arrow_0, = ax.plot([], [], c='g', label=rf'$A$ : Dictionary, $\tau_A = {tau_A} \tau$')
        A_arrows = [A_arrow_0]
        for A in range(self.n_sparse - 1):
            A_arrow, = ax.plot([], [], c='g')
            A_arrows.append(A_arrow)

        print(A_arrows)


        s_n = np.arange(len(s_soln[0]))
        s_h = s_soln[0]
        s_bar = axes[1].bar(s_n, s_h, fc='k')
        axes[1].set_ylim(-10, 10)

        A = A_soln[0]
        print(A.shape)
        x_max, y_max = (s_soln @ A.T).max(0)
        x_min, y_min = (s_soln @ A.T).min(0)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect(1)
        fig.legend(loc='lower right')

        idx_stride = int(len(s_soln) // n_frames)

        def animate(nf):
            idx0 = max(0, (nf - overlap + 1) * idx_stride)
            idx1 = (nf + 1) * idx_stride 

            T = tspan[idx0:idx1]
            y = s_soln[idx0:idx1]
            y = y - np.sign(y)

            ti = T[0]
            tf = T[-1] 

            A = A_soln[idx1]
            x = self.X_shuffled[idx1]
        
            u = (y - self.s0*(np.sign(y))) * (np.abs(y) > self.s0)
            scat_s.set_offsets(u @ A.T)
            scat_s.set_array(np.linspace(0, 1, len(T)))
            scat_s.cmap = plt.cm.get_cmap('Blues')

            scat_x.set_offsets(x)

            for n in range(self.n_sparse):
                a = A[:, n] * 2
                A_arrows[n].set_xdata([0, a[0]])
                A_arrows[n].set_ydata([0, a[1]])

            fig.suptitle(rf'Time: ${ti:.2f} \tau - {tf:.2f} \tau$')

            for i, b in enumerate(s_bar):
                b.set_height(u[0, i])

        anim = animation.FuncAnimation(fig, animate, frames=n_frames-1, interval=100, repeat=True)
        if f_out is not None:
            anim.save(f_out)
        plt.show()

if __name__ == '__main__':

    tau_s = 1e-2
    tau_x = 1
    tau_A = 4e1
    sigma = 1
    T_RANGE = 1e2
    T_STEPS = int(1e4)

    l1 = .5
    l0 = .8

    tspan = th.linspace(0, T_RANGE, T_STEPS)
    #thetas = the.tensor([0, 2 * np.pi/3, -2 * np.pi/3])
    X = th.tensor([
        [np.cos(2 *np.pi/3), np.sin(-2*np.pi/3)],
        #[+1 , +1],
        [np.cos(2 *np.pi/3), np.sin(2 * np.pi/3)],
        [1, 0]
        ]).float()
    X *= 10

    mtsc = MixT_SC(tau_s, tau_x, tau_A, sigma, l1, l0)
    s_soln, A_soln = mtsc.train(X, tspan, n_sparse=3)

    print(s_soln)
    X_evol = X[mtsc.get_x_idx(tspan)]
    dir_out = None
    #dir_out = './figures'
    f_out = './figures/evolution.mp4'
    #f_out = None
    mtsc.save_evolution(s_soln, A_soln, tspan, n_frames=200, f_out=f_out)
