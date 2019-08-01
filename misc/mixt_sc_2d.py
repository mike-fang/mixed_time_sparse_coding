import numpy as np
import sdeint
from tqdm import tqdm
import matplotlib.pylab as plt
from matplotlib import animation
import os.path as osp
from time import time

class MixTSC_2d:
    def __init__(self, tau_s, tau_x, tau_A, sigma, sparsity):
        self.tau_s = tau_s
        self.tau_x = tau_x
        self.tau_A = tau_A
        self.sigma = sigma
        self.sparsity = sparsity
    def train(self, X, tspan, n_sparse):
        self.set_X(X)
        self.tspan = tspan
        self.n_sparse = n_sparse

        A0 = self.init_A()
        s0 = self.init_s()
        sA0 = self.pack_sA(s0, A0)
        print(sA0)

        solution = sdeint.itoint(self.dHdsA, self.dWsA, sA0, tspan)
        #self.solution = sdeint.itoint(self.dHds, self.dWs, s0, tspan)
        self.s_soln = solution[:, :self.n_sparse]
        self.A_soln = solution[:, self.n_sparse:].reshape((-1, self.n_dim, self.n_sparse))
        return self.s_soln, self.A_soln
    def set_X(self, X):
        N, D = X.shape
        self.n_data = N
        self.n_dim = D
        self.X = X
    def init_s(self):
        s0 = np.zeros(self.n_sparse)
        return s0
    def init_A(self):
        theta = np.pi/40
        A = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
            ])
        A = np.random.randn(self.n_dim, self.n_sparse)
        return A
    def get_x_idx(self, t):
        return ( (t//self.tau_x) % self.n_data ).astype(int)
    def dHds(self, s, t):
        x_idx = self.get_x_idx(t)
        x = self.X[x_idx]

        dH_recontr = -self.A.T @ (x - self.A@s) / sigma**2
        dH_sparse = sparsity * np.sign(s)
        return -(dH_recontr + dH_sparse) / self.tau_s
    def dWs(self, s, t):
        return np.eye(self.n_dim) / self.tau_s**0.5
    def dHdsA(self, sA, t):
        s, A = self.unpack_sA(sA)

        # Getting ds
        x_idx = self.get_x_idx(t)
        x = self.X[x_idx]
        #dH_recontr = -A.T @ (x - A@s) * (s**2 > 1) / sigma**2
        s0 = 2
        S = s - np.sign(s) * s0
        S[np.abs(s) < s0] = 0
        dH_recontr = -A.T @ (x - A@S) * (S != 0) / sigma**2
        dH_sparse = sparsity * np.sign(s) 

        ds = -(dH_recontr + dH_sparse) / self.tau_s

        # Getting dA

        A0 = A / np.linalg.norm(A, axis=0)[None, :]
        dA = ((x - A@S)[:, None] @ S[None, :]*1 + 0e-1*(A0-A)) / self.tau_A
        dsA = self.pack_sA(ds, dA)
        return dsA
    def dWsA(self, sA, t):
        dW = np.zeros(self.n_sparse * (1 + self.n_dim)) + 1e-3
        dW[:self.n_dim] = self.tau_s**(-0.5)
        return np.diag(dW)
    def unpack_sA(self, sA):
        s = sA[:self.n_sparse]
        A = sA[self.n_sparse:]
        A = A.reshape((self.n_dim, self.n_sparse))
        return s, A
    def pack_sA(self, s, A):
        A = A.flatten()
        return np.hstack((s, A))
    def plot_traj(self, ax=None, s_soln=None, A_soln=None):
        if s_soln is None:
            s_soln = self.solution

        if ax is None:
            fig, ax = plt.subplots()

        x_idx = self.get_x_idx(tspan)
        y = soln
        A = self.A
        x = X[x_idx]
        ax.scatter(*x.T, c=x_idx, cmap='Set1')
        ax.scatter(*(A @ y.T), s=1, c=x_idx, cmap='Set1')
        ax.set_aspect(1)
    def save_evolution(self, s_soln=None, A_soln=None, tspan=None, n_frames=100, overlap=3, dir_out=None):
        if s_soln is None:
            s_soln = self.s_soln
        if A_soln is None:
            A_soln = self.A_soln
        if tspan is None:
            tspan = self.tspan
        assert len(s_soln) == len(tspan)

        fig, axes = plt.subplots(ncols=2)
        ax = axes[0]

        sx, sy = [], []
        xx, xy = [], []
        scat_s = ax.scatter(sx, sy, s=5, c='b', label=rf'$A \mathbf {{s}}$ : Reconstruction, $\tau_s = {tau_s} \tau$')
        scat_x = ax.scatter(xx, xy, s=20, c='g', label=rf'$\mathbf {{x}}$ : Data, $\tau_x = {tau_x}$')

        a1 = A_soln[0, :, 0] * 5
        a2 = A_soln[0, :, 1] * 5
        _, _, n_sparse = A_soln.shape
        line_A_1, = ax.plot([], [], c='r', label=rf'$A$ : Dictionary, $\tau_A = {tau_A} \tau$')
        line_A_2, = ax.plot([], [], c='r')

        s_n = np.arange(len(s_soln[0]))
        s_h = s_soln[0]
        s_bar = axes[1].bar(s_n, s_h)
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

            x_idx = self.get_x_idx(T)
            A = A_soln[idx1]
            x = X[x_idx[-1]]
        
            scat_s.set_offsets(y @ A.T)
            scat_s.set_array(np.linspace(0, 1, len(T)))
            scat_s.cmap = plt.cm.get_cmap('Blues')

            scat_x.set_offsets(x)

            a1 = A[:, 0] * 2
            a2 = A[:, 1] * 2

            line_A_1.set_xdata([0, a1[0]])
            line_A_1.set_ydata([0, a1[1]])
            line_A_2.set_xdata([0, a2[0]])
            line_A_2.set_ydata([0, a2[1]])

            fig.suptitle(rf'Time: ${ti:.2f} \tau - {tf:.2f} \tau$')

            for i, b in enumerate(s_bar):
                b.set_height(y[0, i])

        anim = animation.FuncAnimation(fig, animate, frames=n_frames-1, interval=100, repeat=True)
        if dir_out is not None:
            anim.save(osp.join(dir_out, 'evolution.mp4'))
        else:
            plt.show()


tau_s = 1e-2
tau_x = 1
tau_A = 1e2
sigma = 2
sparsity = .1

mtsc = MixTSC_2d(tau_s, tau_x, tau_A, sigma, sparsity)

tspan = np.linspace(0, 1e2, int(1e4))

s0 = np.zeros(2)
X = np.array([
    [-1 , -1],
    [+1 , +1],
    [-1 , +1],
    [+1 , -1]
    ])
X *= 10

t0 = time()
mtsc.train(X, tspan, n_sparse=2)
print(time() -t0)
assert False
#mtsc.save_evolution(dir_out='./figures/mtsc2d')
mtsc.save_evolution(dir_out=None, n_frames=200)
