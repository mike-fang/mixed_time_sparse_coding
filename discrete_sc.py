import numpy as np
import matplotlib.pylab as plt
from matplotlib import animation
from helpers import *
from tqdm import tqdm
from visualization import show_img_evo
#from mixed_t_sc

class Energy_L1:
    def __init__(self, sigma, l1, positive=False):
        self.sigma = sigma
        self.l1 = l1
        self.positive = positive
    def ds(self, s, x, A):
        sign_s = np.sign(s)
        dE_recon = (A.T @ (A @ s - x) / self.sigma**2) 
        if self.positive:
            dE_sparse = self.l1 * ( (s > 0) - 10*(s < 0) )
        else:
            dE_sparse = self.l1 * sign_s
        return dE_recon + dE_sparse
    def dA(self, s, x, A):
        return (A @ s - x)[:, None] @ s[None, :] / self.sigma**2
    def __repr__(self):
        desc = f'''Energy_L1
l1 : {self.l1}
positive : {self.positive}
        '''
        return desc

class DiscreteSC:
    def __init__(self, n_dim, n_sparse, energy, im_shape=None, **params):
        self.n_dim = n_dim
        self.n_sparse = n_sparse
        self.energy = energy
        self.params = params
        for k, v in params.items():
            setattr(self, k, v)
    def init_A(self):
        A = np.random.normal(0, 0.4, size=(self.n_dim, self.n_sparse))
        #A = np.eye(2)
        return A
    def init_s(self):
        s = np.zeros(self.n_sparse)
        #s = np.array((1, 0.))
        return s
    def train(self, loader, n_iter, max_iter_s=100, eps=1e-5):
        A = self.init_A()
        A_soln = np.zeros((n_iter, self.n_dim, self.n_sparse))
        s_soln = np.zeros((n_iter, loader.n_batch, self.n_sparse))
        X_soln = np.zeros((n_iter, loader.n_batch, self.n_dim))

        # Init batch counter
        loader.reset()
        # A update loop
        for n in tqdm(range(n_iter)):
            X_batch = loader.get_batch()
            X_soln[n] = X_batch
            s = self.init_s()
            # Data loop
            for n_x, x in enumerate(X_batch):
                ds = 0
                # Find MAP of s
                for i in range(max_iter_s):
                    ds = - self.eta_s * self.energy.ds(s, x, A)
                    if (ds**2).mean()**0.5 < eps:
                        break
                    s += ds
                s_soln[n, n_x] = s
                A += (-self.eta_A * self.energy.dA(s, x, A))/loader.n_batch
            # Normalize A
            A /= np.linalg.norm(A, axis=0)
            A_soln[n] = A

        # Get visible sparse activation

        solns = {
                'A' : A_soln,
                'S' : s_soln,
                'X' : X_soln,
                'T' : np.arange(n_iter)
                }
        return solns
    def show_evolution(self, soln, f_out=None):
        fig, axes = plt.subplots(ncols=2, figsize=(14, 6))
        ax = axes[0]

        # Create scatter plots for r and x
        rx, ry = [], []
        xx, xy = [], []
        scat_r = ax.scatter(rx, ry, s=5, c='b', label=rf'$A \mathbf {{s}}$ : Reconstruction, $\eta_s = {self.eta_s}$')
        scat_x = ax.scatter(xx, xy, s=50, c='r', label=rf'$\mathbf {{x}}$ : Data.')

        # Create arrows for tracking A
        A_arrow_0, = ax.plot([], [], c='g', label=rf'$A$ : Dictionary, $\eta_A = {self.eta_A}$')
        A_arrows = [A_arrow_0]
        for A in range(self.n_sparse - 1):
            A_arrow, = ax.plot([], [], c='g')
            A_arrows.append(A_arrow)

        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect(1)
        fig.legend(loc='lower right')

        # Create bar plot for sparse elements
        if False:
            s_n = np.arange(len(soln['S'][0]))
            s_h = soln['S'][0]
            s_bar = axes[1].bar(s_n, s_h, fc='k')
            axes[1].set_ylim(-10, 10)
            axes[1].set_xticks(np.arange(self.n_sparse))

        def animate(k):
            A = soln['A'][k]
            S = soln['S'][k]
            x = soln['X'][k]

            scat_r.set_offsets(S @ A.T)
            scat_r.cmap = plt.cm.get_cmap('Blues')

            scat_x.set_offsets(x)

            fig.suptitle(rf'Iteration {k}')
            for k in range(self.n_sparse):
                a = A[:, k] * 2
                A_arrows[k].set_xdata([0, a[0]])
                A_arrows[k].set_ydata([0, a[1]])


        anim = animation.FuncAnimation(fig, animate, frames=len(soln['A']), interval=100, repeat=True)
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
    n_dim = 2
    n_sparse = 3
    params = {
        'eta_s': 1e-2,
        'eta_A': 1e-2
            }

    # Define energy
    sigma = 1
    l1 = .5
    energy = Energy_L1(sigma, l1, positive=True)

    # Build data loader
    theta = (np.linspace(0, 2*np.pi, n_sparse, endpoint=False))
    cos = np.cos(theta)
    sin = np.sin(theta)
    X = np.hstack((cos[:,None], sin[:,None]))
    X *= 10
    n_batch = 3
    loader = Loader(X, n_batch)

    dsc = DiscreteSC(n_dim, n_sparse, energy, **params)
    #dsc = DiscreteSC(n_dim, n_sparse, eta_A, eta_s, n_batch, l0, l1, sigma, positive=True)
    solns_dict = dsc.train(loader, n_iter=100, max_iter_s=int(1e2))

    dsc.show_evolution(solns_dict)
