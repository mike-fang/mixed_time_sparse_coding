from ctsc import *
from lca import load_lca_solver
import numpy as np
import torch as th
import h5py
import matplotlib.pylab as plt
import seaborn as sns

class SolnAnalysis:
    def __init__(self, dir_path, lca=False):
        self.dir_path = dir_path
        self.lca = lca
        with open(os.path.join(dir_path, 'params.yaml'), 'r') as f:
            params = yaml.safe_load(f)
            self.params = params
            self.model_params = params['model_params']
            self.solver_params = params['solver_params']
        try:
            self.sigma = params['model_params']['sigma']
        except:
            self.sigma = 1
        if lca:
            self.u0 = params['model_params']['u0']
            self.tau_x = params['solver_params']['n_s']
            self.l1 = 1
        else:
            self.l1 = params['model_params']['l1']
            self.pi = params['model_params']['pi']
            self.tau_x = params['solver_params']['tau_x']
        self.set_soln()
    def set_soln(self, skip=1, offset=0, batch_idx=None):
        offset = offset % skip
        soln = h5py.File(os.path.join(self.dir_path, 'soln.h5'))
        self.soln = {}
        for n in soln:
            self.soln[n] = soln[n][offset::skip]
            if batch_idx is not None:
                if n == 't':
                    pass
                elif n in ['r', 'x']:
                    self.soln[n] = self.soln[n][:, batch_idx:batch_idx+1, :]
                else:
                    self.soln[n] = self.soln[n][:, :, batch_idx:batch_idx+1]
        self.time = self.soln['t']
        self.X = self.soln['x']
        self.S = self.soln['s']
        self.U = self.soln['u']
        self.R = self.soln['r']
        self.A = self.soln['A']

        self.n_t, self.n_batch, self.n_dim = self.X.shape
        _, self.n_dict, _ = self.S.shape
    def psnr(self):
        X_soln = self.soln['x']
        R_soln = self.soln['r']
        mse = np.mean((X_soln - R_soln)**2, axis=1)
        X_max = np.max(X_soln, axis=1)
        return 20 * np.log10(X_max + 1e-9) - 10 * np.log10(mse + 1e-9)
    def mse(self, mean=False):
        X_soln = self.soln['x']
        R_soln = self.soln['r']
        mse = np.mean((X_soln - R_soln)**2, axis=1)
        if mean:
            mse = np.mean(mse, axis=1)
        return mse
    def energy(self, X=None, R=None, U=None, mean=True):
        if X is None:
            X = self.X
        if R is None:
            R = self.R
        if U is None:
            U = self.U
        recon_err = 0.5 * np.mean((X - R)**2, axis=2)
        sparse_loss = np.mean(np.abs(U), axis=1)
        energy = recon_err / self.sigma**2 + self.l1 * sparse_loss
        if mean:
            energy = np.mean(energy, axis=1)
        return energy
    def diff(self):
        X_soln = self.soln['x']
        R_soln = self.soln['r']
        return R_soln - X_soln
    def mean_nz(self, smoothing=1, thresh=0):
        S_soln = self.soln['s'][:]
        non_zero = np.abs(S_soln) > thresh
        mean_nz = non_zero.mean(axis=(1, 2))
        print(np.abs(S_soln).mean())
        smoothing = int(smoothing)
        if smoothing > 1:
            kernel = np.ones(smoothing) / smoothing
            mean_nz = np.convolve(mean_nz, kernel, mode='same')
            mean_nz[-smoothing:] = mean_nz[-smoothing]
            mean_nz[:smoothing] = mean_nz[smoothing]
        return mean_nz
    def plot_nz_hist(self, title='', last_frac=0.1, s_max=3, eps_s=1e-5, log=True, n_bins=100, ylim=None):
        l1 = self.l1

        S_soln = self.soln['s'][:]
        N_S, _, _ = S_soln.shape
        print(S_soln.shape)
        S_converged = S_soln[-int(N_S*last_frac):].flatten()

        l0_sparsity = (S_converged < eps_s).mean()

        bins = np.linspace(eps_s, s_max / l1)
        #bins[1] = eps_s
        #bins = np.insert(bins, 1, eps_s)
        plt.hist(S_converged.flatten(), bins=bins, density=True, fc='grey', ec='black', label='Prob. Distr.')
        prob_expected = l1 * np.exp(-l1 * (bins))
        plt.plot(bins, prob_expected, 'r--', label=r'$P_S(s) = \lambda e^{- \lambda \cdot s}$')
        if log:
            plt.yscale('log')
        if ylim == 'auto':
            plt.ylim(0, 1.2 * l1)
        else:
            plt.ylim(ylim)
        plt.xlim(eps_s, bins[-1])
        plt.ylabel(rf'Distr. of Nonzero Coefficients')
        plt.xlabel(rf'Coeff. Value ($s$); Emperical $P(s > \epsilon_s) = {1-l0_sparsity:.2f}$')

        #title += rf' $(\lambda_1 = {l1:.2f})$'
        plt.title(title)
        plt.legend()
    def show_hist_evo(self, t_inter, thresh = 1e-1):
        n_inter = int(self.n_t // t_inter)
        fig, axes = plt.subplots(n_inter, figsize=(6, 8))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        bins = np.arange(self.n_dict + 1.5) - 0.5
        for n in range(n_inter):
            ax = axes[n]
            t0 = t_inter * n
            t1 = t_inter * (n + 1)
            soln_inter = self.S[t0:t1]
            nonzero = np.abs(soln_inter) > thresh
            n_nonzero = np.sum(nonzero, axis=(1))
            ax.hist(n_nonzero.flatten(), bins=bins, density=True, fc='grey', ec='k')
            ax.set_xticks(bins + 0.5)
    def zero_coeffs(self, thresh):
        S = self.S
        A = self.A

        where_thresh = np.abs(S) < thresh
        Sz = S.copy()
        Sz[where_thresh] = 0
        Rz = np.transpose(A @ Sz, axes=(0, 2, 1))
        E = self.energy(mean=False)
        Ez = self.energy(R=Rz, U=Sz, mean=False)
        where_lower = Ez < E
        SzT = np.transpose(Sz, axes=(0, 2, 1))
        ST = np.transpose(S, axes=(0, 2, 1))
        ST[where_lower] = SzT[where_lower]
        self.R = np.transpose(A @ S, axes=(0, 2, 1))

if __name__ == '__main__':
    dir_path = get_timestamped_dir(load=True, base_dir='bars_dsc')
    analysis = SolnAnalysis(dir_path)
    #print(analysis.solver.tau_x)
    time = analysis.soln['t'][:]
    energy, recon = analysis.energy()
    plt.plot(analysis.soln['t'], recon, 'g')
    plt.yscale('log')

    dir_path = get_timestamped_dir(load=True, base_dir='bars_0T')
    analysis = SolnAnalysis(dir_path)
    #print(analysis.solver.tau_x)
    time = analysis.soln['t'][:]
    energy, recon = analysis.energy()
    plt.plot(analysis.soln['t'], recon, 'r')

    dir_path = get_timestamped_dir(load=True, base_dir='bars_asynch')
    analysis = SolnAnalysis(dir_path)
    #print(analysis.solver.tau_x)
    time = analysis.soln['t'][:]
    energy, recon = analysis.energy()
    plt.plot(analysis.soln['t'], recon, 'b')

    plt.yscale('log')
    plt.show()
