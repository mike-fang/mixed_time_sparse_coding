from ctsc import *
import numpy as np
import torch as th
import h5py
import matplotlib.pylab as plt

class SolnAnalysis:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.soln = h5py.File(os.path.join(dir_path, 'soln.h5'))
        self.solver = load_solver(dir_path)
        self.model = self.solver.model
        self.t_steps = len(self.soln['t'])
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
    def plot_nz_hist(self, last_frac=0.1):
        pi = self.model.pi
        l1 = self.model.l1
        u0 = -np.log(pi) / l1

        S_soln = self.soln['s'][:]
        N_S, _, _ = S_soln.shape
        print(N_S, last_frac)
        S_converged = S_soln[-int(N_S*last_frac):].flatten()

        l0_sparsity = (S_converged == 0).mean()

        hist = np.histogram(S_converged.flatten(), density=True, bins=100)
        trim_idx = (hist[0] < 1e-3).argmax()
        bins = hist[1][:trim_idx]
        #bins[0] = 1e-5
        bins = np.insert(bins, 1, 1e-5)
        plt.hist(S_converged.flatten(), bins=bins, density=True, fc='grey', ec='black', label='Emperical Distr.')
        prob_expected = l1 * np.exp(-l1 * (bins + u0))
        plt.plot(bins, prob_expected, 'r--', label=r'$P_S(s) = \frac{\pi}{\lambda}e^{- \lambda \cdot s}$')
        plt.yscale('log')
        plt.ylim(1e-3, 1)
        plt.xlim(1e-5, bins[-1])
        plt.ylabel('Prob. Density')
        plt.xlabel('Value')
        plt.title('Ditribution of Nonzero Coefficients')
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
