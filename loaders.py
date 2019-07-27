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

        



if __name__ == '__main__':
    H = W = 10
    p = .1
    loader = HVLinesLoader(H, W, 100, p)
    plt.imshow(loader.get_batch())
    plt.show()
