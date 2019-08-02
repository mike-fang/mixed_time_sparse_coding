import numpy as np
import matplotlib.pylab as plt
from matplotlib import animation
from loaders import *
from mixed_t_sc import save_soln
from tqdm import tqdm
from visualization import show_img_evo

class DiscreteSC:
    def __init__(self, n_dim, n_sparse, eta_A, eta_s, n_batch, l0, l1, sigma, positive=False):
        self.n_dim = n_dim
        self.n_sparse = n_sparse
        self.eta_A = eta_A
        self.eta_s = eta_s
        self.n_batch = n_batch

        self.l0 = l0
        self.l1 = l1
        self.sigma = sigma

        self.s0 = -np.log(1 - l0) / l1
        self.positive = True
    def dEds(self, s, x, A):
        sign_s = np.sign(s)
        where_active = (np.abs(s) >= self.s0)
        u = (s - self.s0*(sign_s)) * where_active
        dE_recon = (A.T @ (A @ u - x) / self.sigma**2) * where_active
        if self.positive:
            dE_sparse = self.l1 * ( (s > 0) - 10*(s < 0) )
        else:
            dE_sparse = self.l1 * sign_s
        return dE_recon + dE_sparse
    def dEdA(self, s, x, A):
        u = (s - self.s0*(np.sign(s))) * (np.abs(s) > self.s0)
        return (A @ u - x)[:, None] @ u[None, :] / self.sigma**2
    def init_loader(self, X):
        self.X = np.random.permutation(X)
        self.batch_idx = 0
    def get_batch(self):
        batch_end = self.batch_idx + self.n_batch
        batch = self.X[self.batch_idx:batch_end]

        if batch_end >= len(self.X):
            batch_end %= len(self.X)
            self.init_loader(self.X)
            batch = np.vstack((batch, self.X[:batch_end]))
        self.batch_idx = batch_end
        return batch
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
        s_soln = np.zeros((n_iter, self.n_batch, self.n_sparse))
        X_soln = np.zeros((n_iter, self.n_batch, self.n_dim))

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
                    ds = - self.eta_s * self.dEds(s, x, A)
                    if (ds**2).mean()**0.5 < eps:
                        break
                    s += ds
                s_soln[n, n_x] = s
                A += (-self.eta_A * self.dEdA(s, x, A))/self.n_batch
            # Normalize A
            A /= np.linalg.norm(A, axis=0)
            A_soln[n] = A

        # Get visible sparse activation
        sign_s = np.sign(s_soln)
        where_active = (np.abs(s_soln) >= self.s0)
        u_soln = (s_soln - self.s0*(sign_s)) * where_active

        solns = {
                'A' : A_soln,
                'S' : u_soln,
                'X' : X_soln
                }
        return solns
    def show_evolution(self, soln, f_out=None):
        fig, axes = plt.subplots(ncols=2, figsize=(14, 6))
        ax = axes[0]

        # Create scatter plots for r and x
        rx, ry = [], []
        xx, xy = [], []
        scat_r = ax.scatter(rx, ry, s=5, c='b', label=rf'$A \mathbf {{s}}$ : Reconstruction, $\eta_s = {self.eta_s}$')
        scat_x = ax.scatter(xx, xy, s=50, c='r', label=rf'$\mathbf {{x}}$ : Data, n_batch = {self.n_batch}')

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

if __name__ == '__main__':
    n_dim = 2
    n_sparse = 3
    n_batch = 3
    eta_A = 1e-1
    eta_s = 1e-1
    l0 = .0
    l1 = .5
    sigma = 1

    dsc = DiscreteSC(n_dim, n_sparse, eta_A, eta_s, n_batch, l0, l1, sigma, positive=True)

    theta = (np.linspace(0, 2*np.pi, n_sparse, endpoint=False))
    cos = np.cos(theta)
    sin = np.sin(theta)
    X = np.hstack((cos[:,None], sin[:,None]))
    X *= 10
    #X = np.zeros((1, 2))

    loader = Loader(X, n_batch)
    solns_dict = dsc.train(loader, n_iter=100, max_iter_s=int(1e2))

    dsc.show_evolution(solns_dict)

