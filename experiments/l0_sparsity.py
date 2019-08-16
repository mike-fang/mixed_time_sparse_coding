import sys
sys.path.append('..')
from helpers import *
from scipy.stats import binom

def eval_sparsity(S, n_sparse, idx=None):
    sparsity = (S > 0).sum(axis=2)
    sparsity = sparsity[idx].flatten()
    count = [
            (sparsity == n).mean() for n in range(n_sparse + 1)
            ]
    mean = sparsity.mean()/n_sparse
    return mean, count

# Load dsc soln
model, soln = load_model('../results/hv_dsc_4x4')
n_sparse = model.n_sparse
S_dsc = soln['S']

# Load mtsc soln
model, soln = load_model('../results/hv_mtsc_4x4')
n_sparse = model.n_sparse
S_mtsc = soln['S']

# Get mean and sparse count
idx = slice(-200, None, None)
dsc_sparse_mean, dsc_sparse_count = eval_sparsity(S_dsc, n_sparse, idx)
mtsc_sparse_mean, mtsc_sparse_count = eval_sparsity(S_mtsc, n_sparse, idx)


# Get ML binom_pmf
dsc_pmf = binom.pmf(np.arange(n_sparse + 1), n=n_sparse, p=dsc_sparse_mean)
mtsc_pmf = binom.pmf(np.arange(n_sparse + 1), n=n_sparse, p=mtsc_sparse_mean)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
ax_d, ax_mt = axes


ax_d.bar(range(n_sparse + 1), dsc_sparse_count, fc=(0.8,)*3, ec='k', label='Non-zero activity')
ax_d.plot(np.arange(n_sparse + 1), dsc_pmf, 'ro-', label=fr'ML binomial fit: $p = {dsc_sparse_mean:.3f}$')
ax_d.set_title('Discrete; Point Sample; L1 Energy')
ax_d.set_xticks(range(n_sparse + 1))
ax_d.legend()

ax_mt.bar(range(n_sparse + 1), mtsc_sparse_count, fc=(0.8,)*3, ec='k', label='Non-zero activity')
ax_mt.plot(np.arange(n_sparse + 1), mtsc_pmf, 'ro-', label=fr'ML binomial fit: $p = {mtsc_sparse_mean:.3f}$')
ax_mt.set_title('Mixed Time Analog Sampling; L0 Energy')
ax_mt.set_xticks(range(n_sparse + 1))
ax_mt.legend()

ax_d.set_xlabel('Number of active coefficients')
ax_mt.set_xlabel('Number of active coefficients')

plt.tight_layout()
#plt.savefig('../figures/sparsity_compare.pdf')
plt.show()

