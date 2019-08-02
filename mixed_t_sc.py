import numpy as np
import matplotlib.pylab as plt
from matplotlib import animation
from time import time
from loaders import Loader, Solutions
import h5py
import os.path
from time import time
from tqdm import tqdm

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
def save_soln(solns, im_shape=None, f_name=None, overwrite=False):
    if f_name is None:
        # Output to temp file if none specified
        time_stamp = f'{time():.0f}'
        f_name = os.path.join(FILE_DIR, 'results', 'tmp', time_stamp)
    if os.path.isfile(f_name) and overwrite:
        os.unlink(f_name)

    A = solns['A'][:]
    S = solns['S'][:]
    X = solns['X'][:]
    R = np.einsum('ijk,ilk->ijl', S, A)
    n_f, n_d, n_s = A.shape
    _, n_b, _ = X.shape

    db = h5py.File(f_name)
    db_A = db.create_dataset('A', data=A)
    db_S = db.create_dataset('S', data=S)
    db_X = db.create_dataset('X', data=X)
    db_R = db.create_dataset('R', data=R)
    db.create_dataset('T', data=solns['T'])

    db_A.attrs['transpose'] = (0, 2, 1)
    if im_shape is not None:
        db_A.attrs['reshape'] = (n_f, n_s, *im_shape)
    else:
        db_A.attrs['reshape'] = 'None'

    db_R.attrs['transpose'] = 'None'
    if im_shape is not None:
        db_R.attrs['reshape'] = (n_f, n_b, *im_shape)
    else:
        db_R.attrs['reshape'] = 'None'

    db_X.attrs['transpose'] = 'None'
    if im_shape is not None:
        db_X.attrs['reshape'] = (n_f, n_b, *im_shape)
    else:
        db_X.attrs['reshape'] = 'None'

    return db

class MixedTimeSC:
    def __init__(self, n_dim, n_sparse, tau_s, tau_x, tau_A, l0, l1, sigma, n_batch=3, positive=False):
        self.n_dim = n_dim
        self.n_sparse = n_sparse

        self.tau_s = tau_s
        self.tau_x = tau_x
        self.tau_A = tau_A

        self.l0 = l0
        self.l1 = l1
        self.sigma = sigma
        self.n_batch = n_batch

        self.s0 = -np.log(1 - l0) / l1
        self.positive = positive
    def init_sA(self):
        A = np.random.normal(0, 0.4, size=(self.n_dim, self.n_sparse))
        s = np.zeros((self.n_batch, self.n_sparse))

        if False:
            A = np.eye(self.n_dim)
            s = np.array([1., 4])
        return s, A
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
    def shuffle_X(self, X, tspan):
        n_data = len(X)
        idx = ((tspan // (self.tau_x / n_data))).astype(int)
        max_idx = idx.max()
        rand_idx = np.zeros(max_idx + 1, dtype=int)
        for n in range(0, max_idx+1, n_data):
            try:
                rand_idx[n:n+n_data] = np.random.permutation(n_data)
            except:
                k = len(rand_idx[n:n_data])
                rand_idx[n:n+n_data] = np.random.permutation(n_data)[:k]
        return X[rand_idx[idx]]
    def train(self, loader, tspan, init_sA=None, no_noise=False):
        t0 = time()
        print('Training ...')
        # Start tspan at 0
        tspan -= tspan.min()

        # Definite dt, t_steps
        t_steps = len(tspan)
        dT = tspan[1:] - tspan[:-1]
        
        # Find where to load next X batch
        x_idx = (tspan // self.tau_x).astype(int)
        new_batch = (x_idx[1:] - x_idx[:-1]).astype(bool)

        # Precalculating Wiener process
        dW_s = np.random.normal(loc=0, scale= (2 * dT[:, None, None])**0.5, size=(t_steps - 1, self.n_batch, self.n_sparse)) 
        if no_noise:
            dW_s *= 0

        # Init params and solns
        if init_sA is None:
            S, A = self.init_sA()
        else:
            S, A = init_sA()
        X = loader.get_batch()
        s_soln = np.zeros((t_steps, self.n_batch, self.n_sparse))
        x_soln = np.zeros((t_steps, self.n_batch, self.n_dim))
        A_soln = np.zeros((t_steps, self.n_dim, self.n_sparse))
        s_soln[0] = S
        A_soln[0] = A
        x_soln[0] = X

        # Iterate over time steps
        for i, t in enumerate(tqdm(tspan[1:])):
            # Get new batch 
            if new_batch[i]:
                X = loader.get_batch()
            dt = dT[i]

            #Iterate over batch
            dA = 0
            for j in range(self.n_batch):
                # Calculate gradient
                ds = -self.dEds(S[j], X[j], A) * dt / self.tau_s + dW_s[i, j] / self.tau_s**0.5
                dA += -self.dEdA(S[j], X[j], A) * dt / self.tau_A 

                # Update variables
                S[j] += ds
            # Update changes in A all at once, maybe sequentially with each s might be better
            A += dA / self.n_batch

            # Record values
            s_soln[i+1] = S
            A_soln[i+1] = A
            x_soln[i+1] = X

        sign_s = np.sign(s_soln)
        where_active = (np.abs(s_soln) >= self.s0)
        u_soln = (s_soln - self.s0*(sign_s)) * where_active

        solns = {}
        solns['X'] = x_soln
        #solns['S'] = s_soln
        solns['A'] = A_soln
        solns['T'] = tspan
        solns['S'] = u_soln

        print(f'Training completed in {time() - t0:.2f} seconds')

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

if __name__ == '__main__':
    tau_s = 1e2
    tau_x = 1e4
    tau_A = 1e6
    sigma = .5
    frac = 1
    n_batch = 3

    print(tau_s, tau_x, tau_A)
    T_RANGE = 1e3*frac
    T_STEPS = int(T_RANGE)

    N_DIM = 2
    N_SPARSE = 3

    l1 = .3
    l0 = .5
    tspan = np.linspace(0, T_RANGE, T_STEPS, endpoint=False)

    #theta = [0, 2*np.pi/3, 4*np.pi/3]
    theta = (np.linspace(0, 2*np.pi, N_SPARSE, endpoint=False))
    cos = np.cos(theta)
    sin = np.sin(theta)
    X = np.hstack((cos[:,None], sin[:,None]))
    X *= 10

    loader = Loader(X, n_batch)

    mt_sc = MixedTimeSC(N_DIM, N_SPARSE, tau_s, tau_x, tau_A, l0, l1, sigma, n_batch, positive=True)

    t0 = time()
    solns_dict = mt_sc.train(loader, tspan)
    solns = Solutions(solns_dict)
    solns.save()
    solns2 = Solutions.load()

    #solns_db = save_soln(solns)
    #mt_sc.show_evolution(, n_frames=100)
