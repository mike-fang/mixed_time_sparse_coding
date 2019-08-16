import sys
sys.path.append('..')
from helpers import *
from scipy.stats import binom

def get_err(dir_name):
    model, soln = load_model(dir_name)
    n_sparse = model.n_sparse
    X = soln['X']
    R = soln['R']
    print(soln['T'].shape)
    err = X - R
    MRSE = (err**2).mean(axis=(1, 2))
    return err, MRSE

# Load dsc soln
dsc_dir = '../results/hv_dsc_4x4'
mtsc_dir = '../results/hv_mtsc_4x4'
dsc_err, dsc_MRSE = get_err(dsc_dir)
mtsc_err, mtsc_MRSE = get_err(mtsc_dir)

err = dsc_err[-2:].flatten()
#err = mtsc_err[:200].flatten()
plt.hist(err, range=(-.4, .4), bins=100)
plt.show()

assert False
normed_t_dsc = np.arange(len(dsc_MRSE)) / len(dsc_MRSE)
normed_t_mtsc = np.arange(len(mtsc_MRSE)) / len(mtsc_MRSE)

plt.plot(normed_t_mtsc, mtsc_MRSE)
plt.plot(normed_t_dsc, dsc_MRSE)
plt.show()
