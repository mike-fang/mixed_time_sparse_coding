import torch as th
import numpy as np
from math import pi
import torch.nn.functional as F

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
            #batch += np.random.normal(0, self.sigma, size=batch.shape)
            batch += th.FloatTensor(*batch.shape).normal_(0, self.sigma)
        return batch
    def __call__(self):
        return self.get_batch()
    def __repr__(self):
        str_ = '<Data Loader>\n'
        str_ += f'n_batch: {self.n_batch}\n'
        str_ += f'X: {self.X}\n'
        return str_

class StarLoader_(Loader):
    def __init__(self, n_basis, n_batch, A=10, **kwargs):
        self.n_basis = n_basis
        self.A = A
        X = self.get_X()
        super().__init__(X, n_batch, **kwargs)
    def get_X(self):
        theta = (th.linspace(0, 2*pi, self.n_basis+1)[:-1])
        cos = th.cos(theta)
        sin = th.sin(theta)
        X = self.A * th.cat((cos[:,None], sin[:,None]), dim=1)
        return X

class HVLinesLoader:
    def __init__(self, H, W, n_batch, p=0.1, positive=False):
        self.H = H
        self.W = W
        self.im_shape = (H, W)
        self.n_batch = n_batch
        self.p = p
        self.positive = positive

        self.set_bases()
    def reset(self):
        pass
    def get_batch(self, reshape=False):
        S = th.Tensor(self.n_batch, self.W + self.H).bernoulli_(self.p)
        multiplier = th.Tensor(self.n_batch, self.W + self.H).uniform_()
        if self.positive:
            multiplier -= 0.5
        S *= multiplier
        batch = S @ self.bases

        if reshape:
            return batch.reshape((self.n_batch, self.H, self.W))
        else:
            return batch
    def set_bases(self, flatten=True):
        bases = th.zeros((self.H + self.W, self.H, self.H))
        for i in range(self.H):
            bases[i, i] = 1
        for i in range(self.W):
            bases[self.H + i, :, i] = 1
        if flatten:
            self.bases = bases.reshape((self.H + self.W, -1))
        else:
            self.bases = bases
        return self.bases
    def __call__(self):
        return self.get_batch()
    def __repr__(self):
        desc = 'HVLinesLoader\n'
        desc += f'H, W: {self.H}, {self.W}\n'
        desc += f'n_batch: {self.n_batch}\n'
        desc += f'p: {self.p}'
        return desc

class SparseSampler():
    def __init__(self, A, n_batch, pi=0.2, l1=.2, sigma=0, positive=True):
        self.A = A
        self.n_dim, self.n_dict = A.shape
       
        self.n_batch = n_batch
        self.pi = pi
        self.l1 = l1
        self.sigma = sigma
        self.positive = positive
    def get_coeff(self):
        s =  th.FloatTensor(self.n_dict, self.n_batch).exponential_(self.l1)
        if not self.positive:
            s *= (1 - 2 * th.FloatTensor(*s.shape).bernoulli_(0.5))
        s *= th.FloatTensor(*s.shape).bernoulli_(self.pi)
        return s
    def get_batch(self, transposed=True):
        s = self.get_coeff()
        X = self.A @ s
        if self.sigma > 0:
            X += th.FloatTensor(*X.shape).normal_(0, self.sigma)
        if transposed:
            return X.T
        else:
            return X
    def __call__(self):
        return self.get_batch()

class StarLoader(SparseSampler):
    def __init__(self, n_basis, n_batch, **kwargs):
        n_dim = 2
        self.n_basis = n_basis
        self.n_batch = n_batch
        A = self.get_dict()
        super().__init__(A, n_batch, **kwargs)
    def get_dict(self):
        theta = (th.linspace(0, 2*pi, self.n_basis+1)[:-1])
        cos = th.cos(theta)
        sin = th.sin(theta)
        return th.cat((cos[None, :], sin[None, :]), dim=0)

if __name__ == '__main__':
    n_basis = 3

