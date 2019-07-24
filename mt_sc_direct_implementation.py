import numpy as np
import matplotlib.pylab as plt
from matplotlib import animation
from time import time

class MixT_SC:
    def __init__(self, n_dim, n_sparse, tau_s, tau_x, tau_A, l0, l1, sigma):
        self.n_dim = n_dim
        self.n_sparse = n_sparse

        self.tau_s = tau_s
        self.tau_x = tau_x
        self.tau_A = tau_A

        self.l0 = l0
        self.l1 = l1
        self.sigma = sigma

        self.s0 = -np.log(1 - l0) / l1
    def init_sA(self):
        A = np.random.normal(0, 0.4, size=(self.n_dim, self.n_sparse))
        s = np.zeros(self.n_sparse)

        if False:
            A = np.eye(self.n_dim)
            s = np.array([1., 4])
        return s, A
    def dEds(self, s, x, A):
        sign_s = np.sign(s)
        where_active = (np.abs(s) >= self.s0)
        u = (s - self.s0*(sign_s)) * where_active
        dE_recon = (A.T @ (A @ u - x) / self.sigma**2) * where_active
        #dE_sparse = self.l1 * sign_s
        dE_sparse = self.l1 * ( (s > 0) - 10*(s < 0) )
        return dE_recon + dE_sparse
    def dEdA(self, s, x, A):
        u = (s - self.s0*(np.sign(s))) * (np.abs(s) > self.s0)
        return (A @ u - x)[:, None] @ u[None, :] / self.sigma**2
    def shuffle_X(self, X, tspan):
        n_data = len(X)
        idx = (tspan // self.tau_x).astype(int)
        max_idx = idx.max()
        rand_idx = np.random.randint(0, n_data, size=(max_idx + 1))
        return X[rand_idx[idx]]
    def solve(self, X, tspan):
        # Definite dt, t_steps
        t_steps = len(tspan)
        dT = tspan[1:] - tspan[:-1]

        # Shuffle X
        X = self.shuffle_X(X, tspan)

        # Precalculating Wiener process
        dW_s = np.random.normal(loc=0, scale= (2 * dT[:, None])**0.5, size=(t_steps - 1, self.n_sparse)) 
        # Init params and solns
        s, A = self.init_sA()
        s_soln = np.zeros((t_steps, self.n_sparse))
        A_soln = np.zeros((t_steps, self.n_dim, self.n_sparse))
        s_soln[0] = s
        A_soln[0] = A

        # Iterate over time steps
        for i, t in enumerate(tspan):
            if i == 0:
                continue
            x = X[i]
            dt = dT[i-1]

            # Calculate gradient
            ds = -self.dEds(s, x, A) * dt / self.tau_s + dW_s[i-1] / self.tau_s**0.5
            dA = -self.dEdA(s, x, A) * dt / self.tau_A 


            #print(self.dEds(s, x, A))

            coupling_A_x = t % self.tau_x > self.tau_x/4
            # Update variable
            s += ds
            A += dA * coupling_A_x

            # Record values
            s_soln[i] = s
            A_soln[i] = A
        solns = {}
        solns['X'] = X
        solns['s'] = s_soln
        solns['A'] = A_soln
        solns['T'] = tspan
        return solns
    def save_evolution(self, soln, n_frames=100, overlap=3, f_out=None):
        fig, axes = plt.subplots(ncols=2, figsize=(14, 6))
        ax = axes[0]

        # Create scatter plots for s and x
        sx, sy = [], []
        xx, xy = [], []
        scat_s = ax.scatter(sx, sy, s=5, c='b', label=rf'$A \mathbf {{s}}$ : Reconstruction, $\tau_s = {self.tau_s} \tau$')
        scat_x = ax.scatter(xx, xy, s=50, c='r', label=rf'$\mathbf {{x}}$ : Data, $\tau_x = {self.tau_x}$')

        # Create arrows for tracking A
        a1 = soln['A'][0, :, 0] * 5
        a2 = soln['A'][0, :, 1] * 5
        A_arrow_0, = ax.plot([], [], c='g', label=rf'$A$ : Dictionary, $\tau_A = {self.tau_A} \tau$')
        A_arrows = [A_arrow_0]
        for A in range(self.n_sparse - 1):
            A_arrow, = ax.plot([], [], c='g')
            A_arrows.append(A_arrow)

        # Create bar plot for sparse elements
        s_n = np.arange(len(soln['s'][0]))
        s_h = soln['s'][0]
        s_bar = axes[1].bar(s_n, s_h, fc='k')
        axes[1].set_ylim(-10, 10)
        axes[1].set_xticks(np.arange(self.n_sparse))

        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect(1)
        fig.legend(loc='lower right')

        idx_stride = int(len(soln['s']) // n_frames)
        def animate(nf):
            idx0 = max(0, (nf - overlap + 1) * idx_stride)
            idx1 = (nf + 1) * idx_stride 

            T = soln['T'][idx0:idx1]
            y = soln['s'][idx0:idx1]
            y = y - np.sign(y)

            ti = T[0]
            tf = T[-1] 

            A = soln['A'][idx1]
        
            u = (y - self.s0*(np.sign(y))) * (np.abs(y) > self.s0)
            scat_s.set_offsets(u @ A.T)
            scat_s.set_array(np.linspace(0, 1, len(T)))
            scat_s.cmap = plt.cm.get_cmap('Blues')

            x = soln['X'][idx0:idx1]
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
    tau_s = 1e1
    tau_x = 5e2
    tau_A = 1e4
    sigma = 1
    frac = 1
    T_RANGE = 1e5*frac
    T_STEPS = int(T_RANGE)

    N_DIM = 2
    N_SPARSE = 3

    l1 = .5
    l0 = .8
    tspan = np.linspace(0, T_RANGE, T_STEPS, endpoint=False)

    theta = [0, 2*np.pi/3, 4*np.pi/3]
    cos = np.cos(theta)
    sin = np.sin(theta)
    X = np.hstack((cos[:,None], sin[:,None]))
    X *= 10

    mt_sc = MixT_SC(N_DIM, N_SPARSE, tau_s, tau_x, tau_A, l0, l1, sigma)
    
    t0 = time()
    solns = mt_sc.solve(X, tspan)
    print(time() - t0)
    mt_sc.save_evolution(solns)

