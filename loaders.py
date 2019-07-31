import numpy as np
import matplotlib.pylab as plt
import pickle
from glob import glob
import os.path
from time import time

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
class Loader:
    """
    Loader for predetermined data (X), optionally, iid normal noise can be added to data with stdev sigma.

    Attributes:
        X: The fixed input (n_data, n_dim)
        n_batch: The size of each batch retrieved
        sigma: The stdev of the normal error added (default: 0)

    """
    def __init__(self, X, n_batch, sigma=0):
        self.X = X
        self.n_batch = n_batch
        self.sigma = sigma
        self.reset()
    def reset(self):
        self.X = np.random.permutation(self.X)
        self.batch_idx = 0
    def get_batch(self):
        batch_end = self.batch_idx + self.n_batch
        batch = self.X[self.batch_idx:batch_end]

        if batch_end >= len(self.X):
            batch_end %= len(self.X)
            self.reset()
            batch = np.vstack((batch, self.X[:batch_end]))
        self.batch_idx = batch_end
        if self.sigma > 0:
            batch += np.random.normal(0, self.sigma, size=batch.shape)
        return batch

class HVLinesLoader:
    def __init__(self, H, W, n_batch, p=0.1):
        self.H = H
        self.W = W
        self.n_batch = n_batch
        self.p = p

        self.set_bases()
    def get_batch(self, flatten=True):
        batch = np.zeros((self.n_batch, self.W, self.H))
        for img in batch:
            row_vals = np.random.binomial(1, self.p, self.W) * np.random.random(self.W)
            img += row_vals[:, None] * np.ones(self.H)[None, :]
            col_vals = np.random.binomial(1, self.p, self.H) * np.random.random(self.H)
            img += col_vals[None, :] * np.ones(self.W)[:, None]
        if flatten:
            return batch.reshape((self.n_batch, -1))
        else:
            return batch
    def get_batch(self, reshape=False):
        S = np.random.binomial(1, self.p, size=(self.n_batch, self.W + self.H)).astype(float)
        S *= np.random.random(S.shape)
        batch = S @ self.bases

        if reshape:
            return batch.reshape((self.n_batch, self.H, self.W))
        else:
            return batch
    def set_bases(self, flatten=True):
        bases = np.zeros((self.H + self.W, self.H, self.H))
        for i in range(self.H):
            bases[i, i] = 1
        for i in range(self.W):
            bases[self.H + i, :, i] = 1
        if flatten:
            self.bases = bases.reshape((self.H + self.W, -1))
        else:
            self.bases = bases
        return self.bases

class Solutions:
    @classmethod
    def load(cls, f_name=None):
        if f_name is None:
            # Pick the newest tmp file if none given
            tmp_files = glob(os.path.join(FILE_DIR, 'results', 'tmp', '*'))
            tmp_files.sort()
            f_name = tmp_files[-1]
        with open(f_name, 'rb') as f:
            return pickle.load(f)
    def __init__(self, solns, im_shape=None):
        self.parse_dict(solns)
        if im_shape is None:
            self.H = self.W = None
        else:
            self.H, self.W = im_shape
        if im_shape is not None:
            self.reshape_solns()
    def parse_dict(self, solns):
        self.A = solns['A']
        self.S = solns['S']
        self.X = solns['X']
        if 'T' in solns:
            self.T = solns['T']
        else:
            self.T = None
        if 'R' in solns:
            self.R = solns['R']
        else:
            self.R = np.einsum('ijk,ilk->ijl', self.S, self.A)

        self.n_frame, self.n_dim, self.n_sparse = self.A.shape
        _, self.n_batch, _ = self.X.shape
    def reshape_solns(self):
        A = np.transpose(self.A, (0, 2, 1))
        R = self.R
        X = self.X

        self.A_reshaped = A.reshape((self.n_frame, self.n_sparse, self.H, self.W))
        self.R_reshaped = R.reshape((self.n_frame, self.n_batch, self.H, self.W))
        self.X_reshaped = X.reshape((self.n_frame, self.n_batch, self.H, self.W))
        return db
    def save(self, f_name=None, overwrite=False):
        if f_name is None:
            # Output to temp file if none specified
            time_stamp = f'{time():.0f}'
            f_name = os.path.join(FILE_DIR, 'results', 'tmp', time_stamp)
        if os.path.isfile(f_name) and overwrite:
            os.unlink(f_name)
        with open(f_name, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)




if __name__ == '__main__':

    H = W = 10
    p = .1
    loader = HVLinesLoader(H, W, 100, p)
    plt.imshow(loader.get_batch(reshape=True)[0])
    plt.show()
    assert False
    plt.imshow(loader.get_batch())
    plt.show()
