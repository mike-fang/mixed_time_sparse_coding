from ctsc import *
from lca import load_lca_solver
import numpy as np
import torch as th
import h5py
import matplotlib.pylab as plt
import seaborn as sns
from scipy.stats import norm
from visualization import *

class SolnAnalysis:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        with open(os.path.join(dir_path, 'params.yaml'), 'r') as f:
            params = yaml.safe_load(f)
            self.params = params
            self.model_params = params['model_params']
            self.solver_params = params['solver_params']
        try:
            self.sigma = params['model_params']['sigma']
        except:
            self.sigma = 1
        try:
            self.l1 = params['model_params']['l1']
            #self.pi = params['model_params']['pi']
            self.tau_x = params['solver_params']['tau_x']
            self.tau_s = params['solver_params']['tau_u']
            self.tau_A = params['solver_params']['tau_A']
            self.lca = False
        except:
            self.u0 = params['model_params']['u0']
            self.tau_x = params['solver_params']['n_s']
            self.l1 = 1
            self.lca = True
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

        try:
            self.pi = self.soln['pi']
        except:
            pass

        self.n_t, self.n_batch, self.n_dim = self.X.shape
        _, self.n_dict, _ = self.S.shape
    def find_thresh(self, rel_tol=10):
        S = self.S.flatten()
        Q = np.linspace(0, 1, 1000)
        QS = np.quantile(S, Q)
        d_QS = QS[:-1] - QS[1:]
        rel_dQS = np.abs(d_QS) / np.abs(d_QS[:100]).max()
        if False:
            plt.figure()
            plt.plot(rel_dQS)
            plt.yscale('log')
            plt.show()
        Q_thresh = Q[np.argmax(rel_dQS > rel_tol)]
        S_thresh = np.quantile(S, Q_thresh)
        return S_thresh
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
    def err_distr(self, start=0, end=1, skip=1, dim1=0, dim2=1, max_s=2):
        diff = self.R - self.X
        start = int(self.n_t * start)
        end = int(self.n_t * end)
        print(diff.shape)
        D1 = diff[start:end:skip, :, dim1]
        D2 = diff[start:end:skip, :, dim2]
        g = sns.jointplot(D1, D2, kind='kde')
        g.ax_marg_x.set_xlim(-max_s, max_s)
        g.ax_marg_y.set_ylim(-max_s, max_s)
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
    def binned_mean_nz(self, t_bins):
        N_S, _, _ = self.S.shape
        fracs = np.linspace(0, 1, t_bins)
        mnz=[]
        for start, end in zip(fracs[:-1], fracs[1:]):
            mnz.append((self.S[int(start * N_S):int(end * N_S)] != 0).mean())
        return self.time.max() * 0.5 * (fracs[:-1] + fracs[1:]), mnz
    def get_dkl_s(self, start=0, end=1, eps_s=1e-5, s_max=6, n_bins=100):
        l1 = self.l1
        N_S, _, _ = self.S.shape
        S_converged = self.S[int(N_S*start):int(N_S*end)].flatten()

        l0_sparsity = (S_converged < eps_s).mean()

        bins = np.linspace(eps_s, s_max / l1, n_bins)

        q_i, bin_edge = np.histogram(S_converged.flatten(), bins=bins, density=True)
        s_i = (bin_edge[1:] + bin_edge[:-1]) / 2
        ds = bin_edge[1:] - bin_edge[:-1]
        
        q_i += 1e-5

        q_i *= ds

        cdf_p = np.exp(-l1 * bin_edge)
        P_i = -cdf_p[1:] + cdf_p[:-1]
        P_i /= P_i.sum()
        DKL = np.sum(P_i * (np.log(P_i) - np.log(q_i)))
        return DKL
    def dkls_history(self, t_bins, n_bins, s_max=6, label=None):
        fracs = np.linspace(0, 1, t_bins)
        dkls = []
        for start, end in zip(fracs[:-1], fracs[1:]):
            dkls.append(self.get_dkl_s(start=start, s_max=s_max, end=end, n_bins=25))
        dkls = np.array(dkls)

        
        return self.time.max() * 0.5 * (fracs[:-1] + fracs[1:]), dkls
    def get_dkl_x(self, start=0, end=1, eps_s=1e-5, s_max=5, n_bins=100):
        sigma = self.sigma
        X = self.soln['x'][:]
        R = self.soln['r'][:]

        err = R - X
        err_seg = err[int(self.n_t * start):int(self.n_t * end)]
        bins = np.linspace(-s_max * sigma, s_max * sigma, n_bins)
        q_i, bin_edge = np.histogram(err_seg.flatten(), bins=bins, density=True)
        dx = bin_edge[1:] - bin_edge[:-1]
        q_i *= dx
        q_i += 1e-9

        cdf = norm.cdf(bin_edge/sigma)
        p_i = cdf[1:] - cdf[:-1]

        DKL = np.sum(p_i * (np.log(p_i) - np.log(q_i)))
        return DKL
    def dklx_history(self, t_bins, n_bins, s_max=6, label=None):
        fracs = np.linspace(0, 1, t_bins)
        dkls = []
        for start, end in zip(fracs[:-1], fracs[1:]):
            dkls.append(self.get_dkl_x(start=start, s_max=s_max, end=end, n_bins=25))
        dkls = np.array(dkls)
        
        return self.time.max() * 0.5 * (fracs[:-1] + fracs[1:]), dkls
    def corr_mat_s(self, start=0, end=1):
        S = self.S
        S = S[int(self.n_t * start): int(self.n_t * end)]
        S = np.transpose(S, (0, 2, 1))
        S = S.reshape((-1 , self.n_dict))
        return np.corrcoef(S.T)
    def det_corr(self, corr_mat, one_minus=True):
        det_corr = np.linalg.det(corr_mat)**(1/self.n_dim)
        if one_minus:
            det_corr = 1 - det_corr
        return det_corr
    def corr_mat_x(self, start=0, end=1):
        sigma = self.sigma
        X = self.soln['x'][:]
        R = self.soln['r'][:]

        err = R - X
        err_seg = err[int(self.n_t * start):int(self.n_t * end)]
        err_seg = err_seg.reshape((-1, self.n_dim))
        return np.corrcoef(err_seg.T)
    def det_corr_x(self, start=0, end=1, one_minus=True):
        corr_mat = self.corr_mat_x(start, end)
        det_corr = np.linalg.det(corr_mat)**(1/self.n_dim)
        if one_minus:
            det_corr = 1 - det_corr
        if return_mat:
            return det_corr, corr_mat
        else:
            return det_corr
    def corr_x_hist(self, t_bins):
        fracs = np.linspace(0, 1, t_bins)
        det_corr = []
        for start, end in zip(fracs[:-1], fracs[1:]):
            C = self.corr_mat_x(start, end)
            det_corr.append(self.det_corr(C))
            #det_corr.append(self.det_corr_x(start=start, end=end))
        dkls = np.array(det_corr)
        
        return self.time.max() * 0.5 * (fracs[:-1] + fracs[1:]), dkls
    def corr_s_hist(self, t_bins):
        fracs = np.linspace(0, 1, t_bins)
        det_corr = []
        for start, end in zip(fracs[:-1], fracs[1:]):
            C = self.corr_mat_s(start, end)
            det_corr.append(self.det_corr(C))
            #det_corr.append(self.det_corr_x(start=start, end=end))
        dkls = np.array(det_corr)
        
        return self.time.max() * 0.5 * (fracs[:-1] + fracs[1:]), dkls
    def plot_nz_hist(self, title='', start=0, end=1, s_max=3, eps_s=1e-5, log=False, n_bins=100, ylim=None):
        l1 = self.l1
        S_soln = self.soln['s'][:]
        N_S, _, _ = S_soln.shape
        start_idx = int(N_S * start)
        end_idx = int(N_S * end)
        S_converged = S_soln[start_idx:end_idx].flatten()

        l0_sparsity = (S_converged < eps_s).mean()

        bins = np.linspace(eps_s, s_max / l1, n_bins)
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
        #plt.ylabel(rf'Distr. of Nonzero Coefficients; Activity = {1-l0_sparsity:.2f}')
        plt.ylabel(rf'Distr. of positive Coeff.')
        plt.xlabel(rf'Coeff. Value; $P(s_i > 0) = {1-l0_sparsity:.2f}$')

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
    def zero_coeffs(self, thresh=None):
        S = self.S
        A = self.A
        if thresh is None:
            thresh = self.find_thresh()
            print(f'Setting threshold to {thresh:.3f}')
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
    def norm_A(self):
        return np.linalg.norm(self.A, axis=1)
    def plot_dict(self, im_size, ncol, nrow):
        A = self.A[-1]
        plot_dict(A[:, order], im_size, ncol, nrow)
        plt.tight_layout()
        #plt.subplots_adjust(hspace=.1, wspace=.1, left=.05, right=.95, bottom=.05, top=.95)
    def plot_dict_norm(self, alpha=.2):
        plt.ylabel(r'Dict. Element Norm')
        norm_A = np.linalg.norm(self.A, axis=1)
        for n, norm in enumerate(norm_A.T[norm_A[-1].argsort()]):
            q = n / len(norm_A.T)
            plt.plot(self.time, norm, c='k', alpha=alpha)




if __name__ == '__main__':
    dir_path = get_timestamped_dir(load=True, base_dir='bars_lsc')
    analysis = SolnAnalysis(dir_path)
