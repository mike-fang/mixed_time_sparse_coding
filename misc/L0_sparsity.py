from mtsc import *
from solution_saver import *
import os
from scipy.stats import binom

dir_path = get_timestamped_dir(load=True, base_dir='hv_lines')

f_names = glob('./results/hv_lines/*')
f_names.sort()
dir_paths = f_names[-9:]

def plot_sparsity(dir_path, pi, out=True):
    plt.figure(figsize=(12, 8))
    soln_path = os.path.join(dir_path, 'soln.h5')
    soln = Solutions.load(soln_path)
    S = soln['s_data']
    pi_over_t = np.sum(S > 0, axis=1)
    N, n_batch = pi_over_t.shape

    pi_last = pi_over_t[-N//2:].flatten()
    mu = pi_last.mean()/8

    pi = .2
    binomial = binom(8, pi)
    K = np.arange(9)
    P = binomial.pmf(K)


    plt.hist(pi_last, density=True, fc='grey', ec='k', bins=np.arange(10) - .5)
    plt.plot(K, P, 'ro-', label='Expected Distribution')
    plt.title(f'mean = {mu:.2f}; expected = {pi:.2f}')
    plt.legend()
    if out:
        plt.savefig(os.path.join(dir_path, 'sparsity_distr.pdf'))
    plt.show()
