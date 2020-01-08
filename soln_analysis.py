from ctsc import *
from lca import load_lca_solver

import numpy as np
import torch as th
import h5py
import matplotlib.pylab as plt

class SolnAnalysis:
    def __init__(self, dir_path, lca=False):
        self.dir_path = dir_path
        if lca:
            self.solver = load_lca_solver(dir_path)
        else:
            self.solver = load_solver(dir_path)
        self.model = self.solver.model
        self.set_soln()
    def set_soln(self, skip=1, offset=0, batch_idx=None):
        offset = offset % skip
        soln = h5py.File(os.path.join(self.dir_path, 'soln.h5'))
        self.soln = {}
        for n in soln:
            skip = int(skip)
            self.soln[n] = soln[n][offset::skip]
            if batch_idx is not None:
                if n == 't':
                    pass
                elif n in ['r', 'x']:
                    self.soln[n] = self.soln[n][:, batch_idx:batch_idx+1, :]
                else:
                    self.soln[n] = self.soln[n][:, :, batch_idx:batch_idx+1]
    def energy(self, skip=1):
        X_soln = self.soln['x'][::skip]
        U_soln = self.soln['u'][::skip]
        A_soln = self.soln['A'][::skip]
        time = self.soln['t'][::skip]
        energy = np.zeros_like(time)
        recon = np.zeros_like(time)
        for n, (A, u, x) in enumerate(zip(A_soln, U_soln, X_soln)):
            energy[n], recon[n] = self.model.energy(x, A=A, u=u, return_recon=True)
        return time, energy, recon
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
    def plot_nz_hist(self, title='', last_frac=0.1, s_max=3, eps_p=1e-5, log=True, n_bins=100, ylim=None):
        pi = self.model.pi
        l1 = self.model.l1
        u0 = -np.log(pi) / l1

        S_soln = self.soln['s'][:]
        N_S, _, _ = S_soln.shape
        S_converged = S_soln[-int(N_S*last_frac):].flatten()

        l0_sparsity = (S_converged == 0).mean()

        bins = np.linspace(0, s_max / l1)
        bins = np.insert(bins, 1, eps_p)
        plt.hist(S_converged.flatten(), bins=bins, density=True, fc='grey', ec='black', label='Prob. Distr.')
        prob_expected = l1 * np.exp(-l1 * (bins + u0))
        plt.plot(bins, prob_expected, 'r--', label=r'$P_S(s) = \frac{\pi}{\lambda}e^{- \lambda \cdot s}$')
        if log:
            plt.yscale('log')
        if ylim == 'auto':
            plt.ylim(0, 1.2 * l1 * pi)
        else:
            plt.ylim(ylim)
        plt.xlim(eps_p, bins[-1])
        plt.ylabel(rf'Distr. of Nonzero Coefficients')
        plt.xlabel(rf'Coeff. Value ($s$); Emperical $P(s > 0) = {1-l0_sparsity:.2f}$')

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
