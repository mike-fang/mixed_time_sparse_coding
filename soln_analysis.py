from ctsc import *
from lca import load_lca_solver

import numpy as np
import torch as th
import h5py
import matplotlib.pylab as plt


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
    def psnr(self):
        X_soln = self.soln['x']
        R_soln = self.soln['r']
        mse = np.mean((X_soln - R_soln)**2, axis=1)
        X_max = np.max(X_soln, axis=1)
        return 20 * np.log10(X_max + 1e-9) - 10 * np.log10(mse + 1e-9)
    def mse(self):
        X_soln = self.soln['x']
        R_soln = self.soln['r']
        return np.mean((X_soln - R_soln)**2, axis=1)
    def energy(self):
        X_soln = self.soln['x']
        U_soln = self.soln['u']
        R_soln = self.soln['r']

        recon_err = 0.5 * np.mean((X_soln - R_soln)**2, axis=(1, 2))
        sparse_loss = np.mean(np.abs(U_soln), axis=(1, 2))
        energy = recon_err / self.sigma**2 + self.l1 * sparse_loss
        return energy
    def diff(self):
        X_soln = self.soln['x']
        R_soln = self.soln['r']
        return R_soln - X_soln
    def mean_nz(self, smoothing=1):
        S_soln = self.soln['s'][:]
        non_zero = S_soln != 0
        mean_nz = non_zero.mean(axis=(1, 2))
        smoothing = int(smoothing)
        if smoothing > 1:
            kernel = np.ones(smoothing) / smoothing
            mean_nz = np.convolve(mean_nz, kernel, mode='same')
            mean_nz[-smoothing:] = mean_nz[-smoothing]
            mean_nz[:smoothing] = mean_nz[smoothing]
        return mean_nz
    def plot_nz_hist(self, title='', last_frac=0.1, s_max=3, eps_s=1e-5, log=True, n_bins=100, ylim=None):
        pi = self.pi
        l1 = self.l1
        u0 = -np.log(pi) / l1

        S_soln = self.soln['s'][:]
        N_S, _, _ = S_soln.shape
        S_converged = S_soln[-int(N_S*last_frac):].flatten()

        l0_sparsity = (S_converged < eps_s).mean()

        bins = np.linspace(0, s_max / l1)
        bins[1] = eps_s
        #bins = np.insert(bins, 1, eps_s)
        plt.hist(S_converged.flatten(), bins=bins, density=True, fc='grey', ec='black', label='Prob. Distr.')
        prob_expected = l1 * np.exp(-l1 * (bins + u0))
        plt.plot(bins, prob_expected, 'r--', label=r'$P_S(s) = \frac{\pi}{\lambda}e^{- \lambda \cdot s}$')
        if log:
            plt.yscale('log')
        if ylim == 'auto':
            plt.ylim(0, 1.2 * l1 * pi)
        else:
            plt.ylim(ylim)
        plt.xlim(eps_s, bins[-1])
        plt.ylabel(rf'Distr. of Nonzero Coefficients')
        plt.xlabel(rf'Coeff. Value ($s$); Emperical $P(s > \epsilon_s) = {1-l0_sparsity:.2f}$')

        title += rf' $(\pi = {pi:.2f}, \lambda_1 = {l1:.2f})$'
        plt.title(title)
        plt.legend()

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
