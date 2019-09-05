import torch as th
from math import pi

class Loader:
    """
    Loader for predetermined data (X), optionally, iid normal noise can be added to data with stdev sigma.

    Attributes:
        X: The fixed input (n_data, n_dim)
        n_batch: The size of each batch retrieved
        sigma: The stdev of the normal error added (default: 0)

    """
    def __init__(self, X, n_batch, shuffle=True, sigma=0):
        self.X = X
        self.n_batch = n_batch
        self.sigma = sigma
        self.shuffle = shuffle
        self.reset()
    def reset(self):
        if self.shuffle:
            rand_idx = th.randperm(len(self.X))
            self.X = self.X[rand_idx]
        self.batch_idx = 0
    def get_batch(self):
        batch_end = self.batch_idx + self.n_batch
        batch = self.X[self.batch_idx:batch_end]

        if batch_end >= len(self.X):
            batch_end %= len(self.X)
            self.reset()
            #batch = np.vstack((batch, self.X[:batch_end]))
            batch = th.cat((batch, self.X[:batch_end]))
        self.batch_idx = batch_end
        if self.sigma > 0:
            noise 
            #batch += np.random.normal(0, self.sigma, size=batch.shape)
            batch += th.FloatTensor(batch.size).normal_(0, self.sigma)
        return batch
    def __call__(self):
        return self.get_batch()
    def __repr__(self):
        str_ = '<Data Loader>\n'
        str_ += f'n_batch: {self.n_batch}\n'
        str_ += f'X: {self.X}\n'
        return str_

class StarLoader(Loader):
    def __init__(self, n_basis, n_batch, A=10, sigma=0):
        self.n_basis = n_basis
        self.A = A
        X = self.get_X()
        super().__init__(X, n_batch, sigma)
    def get_X(self):
        theta = (th.linspace(0, 2*pi, self.n_basis+1)[:-1])
        cos = th.cos(theta)
        sin = th.sin(theta)
        X = self.A * th.cat((cos[:,None], sin[:,None]), dim=1)
        return X

class HVLinesLoader:
    def __init__(self, H, W, n_batch, p=0.1):
        self.H = H
        self.W = W
        self.im_shape = (H, W)
        self.n_batch = n_batch
        self.p = p

        self.set_bases()
    def reset(self):
        pass
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
    def __repr__(self):
        desc = 'HVLinesLoader\n'
        desc += f'H, W: {self.H}, {self.W}\n'
        desc += f'n_batch: {self.n_batch}\n'
        desc += f'p: {self.p}'
        return desc

if __name__ == '__main__':
    n_basis = 3
    loader = StarLoader(n_basis, 2)

    for _ in range(10):
        print(loader())
