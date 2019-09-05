import torch as th
from torch.nn import Parameter
from torch.nn import Module
import torch.nn.functional as F
import numpy as np
from loaders import StarLoader, Loader
from tqdm import tqdm
import matplotlib.pylab as plt
from matplotlib import animation

class MTParameter(Parameter):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mu = 0
        self.tau = None
        self.temp = 0
        self.momentum = th.zeros(self.shape)
    @property
    def mass(self):
        return self.mu * self.tau**2
    def update_x(self, dt):
        if self.tau in [0, None]:
        # Fixed parameter
            return
        if self.mu == 0:
        # No mass
            self.data += -self.grad/self.tau * dt
            if self.temp > 0:
                dW = th.FloatTensor(self.size()).normal_()
                self.data += self.temp * dW / (dt * self.tau)**0.5
        else:
            self.data += 0.5 * self.momentum * dt / self.mass
    def update_p(self, dt):
        if self.mu == 0:
            return 
        self.momentum += -self.tau * self.momentum * dt / self.mass - self.grad * dt
        if self.temp > 0:
            dW = th.FloatTensor(self.size()).normal_()
            self.momentum += (self.temp * self.tau / dt)**0.5 * dW
     
class MixedTimeSC(Module):
    def __init__(self, n_dim, n_dict, n_batch, tau, mass, T, init=[None], positive=False):
        super().__init__()
        self.n_dim = n_dim
        self.n_dict = n_dict
        self.n_batch = n_batch

        self.tau = tau
        self.mass = mass
        self.T = T
        self.positive = positive

        self.init = init
        self.init_params()
        self.set_properties()
    def init_params(self, init=None):
        self.A = MTParameter(th.Tensor(self.n_dim, self.n_dict))
        self.s = MTParameter(th.Tensor(self.n_dict, self.n_batch))
        self.rho = MTParameter(th.Tensor(1))
        self.l1 = MTParameter(th.Tensor(1))
        self.nu = MTParameter(th.Tensor(1))

        # nu = log(pi / (1-pi)) log odds ratio
        self.reset_params()
    def reset_params(self):
        self.A.data.normal_()
        self.s.data.normal_()
        #self.s.data *= 0
        self.rho.data = th.tensor(1.)
        self.l1.data = th.tensor(1.)
        self.nu.data = th.tensor(1.)
        for n, p in self.named_parameters():
            if n in self.init:
                p.data = th.tensor(self.init[n])
    def set_properties(self):
        for n, p in self.named_parameters():
            if n in self.tau:
                p.tau = self.tau[n]
                if n in self.mass:
                    p.mu = self.mass[n]
                if n in self.T:
                    p.temp = self.T[n]
    @property
    def u(self):
        if True:
            s0 = -self.l1 * th.log(th.sigmoid(-self.nu))
            u = F.relu(self.s - s0) - F.relu(-self.s - s0)
        else:
            u = self.s
        if self.positive:
            beta = 1
            u = F.softplus(beta * u) 
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
    def update_params(self, x, dt):
        for n, p in self.named_parameters():
            if p.mu > 0:
                # Half step x
                p.update_x(dt/2)
        self.zero_grad()
        self.energy(x).backward()
        for n, p in self.named_parameters():
            if p.mu > 0:
                # Update p
                p.update_p(dt)
                # Half step x
                p.update_x(dt/2)
            else:
                p.update_x(dt)
    def train(self, loader, tspan, out_t=None):
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
        x_idx = (tspan // self.tau['x']).astype(int)
        new_batch = (x_idx[1:] - x_idx[:-1]).astype(bool)
        x = loader()
        
        soln = {}
        for n, p in self.named_parameters():
            soln[n] = np.zeros((len(out_t), *p.size()))
        soln['u'] = np.zeros((len(out_t), *self.s.size()))
        soln['x'] = np.zeros((len(out_t), self.n_batch, self.n_dim))
        soln['r'] = np.zeros((len(out_t), self.n_batch, self.n_dim))
        soln['T'] = np.zeros(len(out_t))

        def add_to_soln(i):
            for n, p in self.named_parameters():
                soln[n][i] = p.data
            soln['u'][i] = self.u.data
            soln['x'][i] = x.data
            soln['r'][i] = self.r.data
            soln['T'][i] = out_t[i]

        add_to_soln(0)
        soln_counter = 1
        for n, t in enumerate(tqdm(tspan[1:])):
            if new_batch[n]:
                x = loader()
            dt = dT[n]
            self.update_params(x, dt)

            #Store solution if in out_t
            if t == out_t[soln_counter]:
                add_to_soln(soln_counter)
                soln_counter += 1

        return soln

def show_evolution(soln, n_frames=100, overlap=3, f_out=None):
    X = soln['x']
    S = soln['s']
    t_steps, n_batch, n_dim = X.shape
    assert n_dim == 2
    _, n_dict, _ = S.shape
    idx_stride = int(t_steps / n_frames)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Create scatter plots for s and x
    sx, sy = [], []
    xx, xy = [], []
    scat_s = ax.scatter(sx, sy, s=5, c='b', label=rf'$A \mathbf {{u}}$ : Reconstruction')
    scat_x = ax.scatter(xx, xy, s=50, c='r', label=rf'$\mathbf {{x}}$ : Data')

    # Create arrows for tracking A
    A_arrow_0, = ax.plot([], [], c='g', label=rf'$A$ : Dictionary bases')
    A_arrows = [A_arrow_0]
    for A in range(n_dict):
        A_arrow, = ax.plot([], [], c='g')
        A_arrows.append(A_arrow)

    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect(1)
    fig.legend(loc='lower right')

    def animate(nf):
        idx0 = max(0, (nf - overlap + 1) * idx_stride)
        idx1 = (nf + 1) * idx_stride 

        T = soln['T'][idx0:idx1]
        r = soln['r'][idx0:idx1]
        s = soln['s'][idx0:idx1].reshape((-1, n_dict))

        ti = T[0]
        tf = T[-1] 

        A = soln['A'][idx1]
    
        scat_s.set_offsets(r.reshape((-1, 2)))
        scat_s.set_array(np.linspace(0, 1, len(T)))
        scat_s.cmap = plt.cm.get_cmap('Blues')

        x = soln['x'][idx0:idx1].reshape((-1, n_dim))
        scat_x.set_offsets(x)

        for n in range(n_dict):
            A_arrows[n].set_xdata([0, A[0, n]])
            A_arrows[n].set_ydata([0, A[1, n]])

        fig.suptitle(rf'Time: ${ti:.2f} \tau - {tf:.2f} \tau$')

    anim = animation.FuncAnimation(fig, animate, frames=n_frames-1, interval=100, repeat=True)
    if f_out is not None:
        anim.save(f_out)
    plt.show()

if __name__ == '__main__':
    n_dict = 3
    n_dim = 2
    n_batch = 3
    mass = {
            's' : .00,
            'A' : 0,
            }
    T = {
            's': 1.,
            }
    tau = {
            's': 1e2,
            'x': 1e3,
            'A': 1e5,
            'l1': None,
            'nu': None,
            'rho': None,
            }

    init = {
            'l1' : 1.,
            'nu' : .5,
            'rho' : .5,
            }

    T_RANGE = 1e4
    T_STEPS = int(T_RANGE)
    tspan = np.linspace(0, T_RANGE, T_STEPS, endpoint=False)

    mtsc = MixedTimeSC(n_dim, n_dict, n_batch, tau, mass, T, init=init, positive=True)
    #loader = Loader(10 * th.eye(2), n_batch, shuffle=False)
    loader = StarLoader(3, n_batch)
    soln = mtsc.train(loader, tspan, None)
    show_evolution(soln)
