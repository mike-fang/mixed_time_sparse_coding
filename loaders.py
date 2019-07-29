import numpy as np
import matplotlib.pylab as plt

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



        



if __name__ == '__main__':

    H = W = 10
    p = .1
    loader = HVLinesLoader(H, W, 100, p)
    plt.imshow(loader.get_batch(reshape=True)[0])
    plt.show()
    assert False
    plt.imshow(loader.get_batch())
    plt.show()
