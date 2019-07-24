import torch as th
from torch.nn.parameter import Parameter
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt
from matplotlib import animation
import os.path as osp
from time import time
from mt_sde_solver import *

def L2_reconstr_loss(A, s, x, sigma):
    return 0.5 * th.norm(x - A @ s)**2 / sigma**2 
def E_SC_L0(params, t):
    A = params['A']
    s = params['s']
    x = params['x']
    sigma = params['sigma']
    l1 = params['l1']
    #return L2_reconstr_loss(A, s, x, sigma)
    if False:
        l0 = params['l0']
        s0 = -th.log(1 - l0) / l1
        u = (s - s0*(th.sign(s))) * (th.abs(s) > s0).float()
    return 0 * L2_reconstr_loss(A, s, x, sigma) + l1 * th.norm(s, p=1)
class MT_SC:
    def __init__(self, n_dim, n_sparse, tau_s, tau_x, tau_A, l0, l1, sigma):
        self.n_dim = n_dim
        self.n_sparse = n_sparse

        self.tau_s = tau_s
        self.tau_x = tau_x
        self.tau_A = tau_A

        self.l0 = l0
        self.l1 = l1
        self.sigma = sigma

        self.init_params()
        self.solver = MixT_SDE(self.params, E_SC_L0)
    def init_params(self):
        A = get_param((self.n_dim, self.n_sparse), tau=self.tau_A, T=0)
        s = get_param((self.n_sparse), tau=self.tau_s, T=1)
        x = get_param((self.n_dim), tau=self.tau_x)

        A.data.normal_()
        A.data *= 4

        if True:
            A.freq = 0
            A.requires_grad = False
            A.data = th.eye(2)
            s.data = th.Tensor((1, 2))
            s.T = 1
            x.T = 0
            x.freq = 0
            x.data *= 0

        l0 = get_param(1, tau=None, init=self.l0)
        l1 = get_param(1, tau=None, init=self.l1)
        sigma = get_param(1, tau=None, init=self.sigma)
        self.params = {
                'A' : A,
                's' : s,
                'x' : x,
                'l0' : l0,
                'l1' : l1,
                'sigma' : sigma
                }
    def solve_X(self, X, tspan):
        return self.solver.solve('x', X, tspan)
    def save_evolution(self, param_evol, n_frames=100, overlap=3, f_out=None):

        x_soln = param_evol['x'].data.numpy() 
        s_soln = param_evol['s'].data.numpy()
        A_soln = param_evol['A'].data.numpy()
        tspan =  param_evol['tspan'].data.numpy()

        tau_s = self.params['s'].freq ** (-1)
        tau_A = self.params['A'].freq ** (-1)
        tau_x = self.params['x'].freq ** (-1)

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

        s_n = np.arange(len(s_soln[0]))
        s_h = s_soln[0]
        s_bar = axes[1].bar(s_n, s_h, fc='k')
        axes[1].set_ylim(-10, 10)

        A = A_soln[0]
        x_max, y_max = (s_soln @ A.T).max(0)
        x_min, y_min = (s_soln @ A.T).min(0)
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
        
            #u = (y - self.s0*(np.sign(y))) * (np.abs(y) > self.s0)
            u = y
            scat_s.set_offsets(u @ A.T)
            #scat_s.set_array(np.linspace(0, 1, len(T)))
            #scat_s.cmap = plt.cm.get_cmap('Blues')

            x = x_soln[idx0:idx1]
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
    hyper_params = {}
    hyper_params['n_dim'] = 2
    hyper_params['n_sparse'] = 2
    hyper_params['tau_s'] = 1e-2
    hyper_params['tau_x'] = 1
    hyper_params['tau_A'] = 4e1
    hyper_params['l0'] = .5
    hyper_params['l1'] = 1
    hyper_params['sigma'] = 1

    mtsc = MT_SC(**hyper_params)

    frac = 0.05
    T_RANGE = 1e0 * frac
    T_STEPS = int(1e4 * frac)

    tspan = th.linspace(0, T_RANGE, T_STEPS + 1)[:-1]
    print(tspan[1] - tspan[0])

    X = th.tensor([
        [np.cos(2 *np.pi/3), np.sin(-2*np.pi/3)],
        [np.cos(2 *np.pi/3), np.sin(2 * np.pi/3)],
        [1, 0]
        ]).float()
    X = th.tensor([
        [0, 0]
        ]).float()
    X *= 10
    soln = mtsc.solve_X(X, tspan)
    print(soln['s'])
    plt.hist(soln['s'][:, 0], bins=20)
    #plt.scatter(*(soln['s']).data.numpy().T, c=np.arange(len(tspan)))
    plt.show()
    #mtsc.save_evolution(soln)
