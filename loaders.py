import torch as th
import numpy as np
from math import pi
import torch.nn.functional as F
import h5py
from scipy.io import loadmat
import matplotlib.pylab as plt
from time import time
from visualization import plot_dict

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
    def __call__(self, n_batch):
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

class ZIELoader:
    def __init__(self, bases, n_batch, pi=0.1, positive=True, numpy=False, sigma=1, l1=1):
        self.l1 = l1
        self.n_batch = n_batch
        self.pi = pi
        self.positive = positive
        self.numpy = numpy
        self.sigma = sigma

        if isinstance(bases, tuple):
            bases = th.eye(*bases)
        _, self.n_dim = bases.shape
        self.bases = bases
    def get_batch(self, n_batch=None):
        if n_batch is None:
            n_batch = self.n_batch
        S = self.get_coeff(n_batch=n_batch)

        batch = S @ self.bases
        noise = th.Tensor(batch.shape)
        noise.normal_()
        batch += noise * self.sigma

        if self.numpy:
            batch = np.array(batch)
        return batch
    def get_coeff(self, n_batch=None):
        if n_batch is None:
            n_batch = self.n_batch
        S = th.Tensor(n_batch, self.n_dict).bernoulli_(self.pi)
        coeff = np.abs(np.random.laplace(0, scale=1/self.l1, size=(n_batch, self.n_dict)))
        multiplier = th.tensor(coeff)
        S *= multiplier
        if not self.positive:
            flip = th.Tensor(n_batch, self.n_dict).bernoulli_(0.5) * 2 - 1
            S *= flip
        return S
    def set_bases(self, flatten=True):
        bases = th.zeros((self.H + self.W, self.H, self.W))
        for i in range(self.H):
            bases[i, i] = self.W**(-0.5)
        for i in range(self.W):
            bases[self.H + i, :, i] = self.H**(-0.5)

        if flatten:
            self.bases = bases.reshape((self.H + self.W, -1))
        else:
            self.bases = bases
        return self.bases
    @property
    def n_dict(self):
        return len(self.bases)
    def __call__(self, n_batch=None):
        return self.get_batch(n_batch=n_batch)

class BarsLoader:
    def __init__(self, H, W, n_batch, p=0.1, positive=True, test=False, numpy=False, sigma=1, l1=1):
        self.l1 = l1
        self.H = H
        self.W = W
        self.im_shape = (H, W)
        self.n_batch = n_batch
        self.pi = p
        self.positive = positive
        self.test = test
        self.numpy = numpy
        self.sigma = sigma

        self.set_bases()
    def reset(self):
        pass
    def get_batch(self, reshape=False, n_batch=None, test=False):
        if self.test:
            test = True
        if n_batch is None:
            n_batch = self.n_batch
        if test:
            d_min = min(n_batch, self.n_dict)
            S *= 0
            S[:d_min, :d_min] = th.eye(d_min)
        else:
            S = self.get_coeff(n_batch=n_batch)

        batch = S @ self.bases
        noise = th.Tensor(batch.shape)
        noise.normal_()
        batch += noise * self.sigma

        if self.numpy:
            batch = np.array(batch)

        if reshape:
            return batch.reshape((n_batch, self.H, self.W))
        else:
            return batch
    def get_coeff(self, n_batch=None):
        if n_batch is None:
            n_batch = self.n_batch
        S = th.Tensor(n_batch, self.n_dict).bernoulli_(self.pi)
        coeff = np.abs(np.random.laplace(0, scale=1/self.l1, size=(n_batch, self.n_dict)))
        multiplier = th.tensor(coeff)
        S *= multiplier
        if not self.positive:
            flip = th.Tensor(n_batch, self.n_dict).bernoulli_(0.5) * 2 - 1
            S *= flip
        return S
    def set_bases(self, flatten=True):
        bases = th.zeros((self.H + self.W, self.H, self.W))
        for i in range(self.H):
            bases[i, i] = self.W**(-0.5)
        for i in range(self.W):
            bases[self.H + i, :, i] = self.H**(-0.5)

        if flatten:
            self.bases = bases.reshape((self.H + self.W, -1))
        else:
            self.bases = bases
        return self.bases
    @property
    def n_dict(self):
        return len(self.bases)
    def __call__(self, n_batch=None):
        return self.get_batch(n_batch=n_batch)
    def __repr__(self):
        desc = 'HVLinesLoader\n'
        desc += f'H, W: {self.H}, {self.W}\n'
        desc += f'n_batch: {self.n_batch}\n'
        desc += f'p: {self.pi}'
        return desc

class DominosLoader(BarsLoader):
    def __init__(self, H, W, n_batch, p=0.1, positive=True, test=False, numpy=False, sigma=1, l1=1):
        super().__init__(H, W, n_batch, p, positive, test, numpy, sigma, l1)
        self.set_bases()
    def set_bases(self, flatten=True):
        #bases = th.zeros((self.n_dict, self.H, self.W))
        bases = []
        for x in range(self.W):
            for y in range(self.H):
                if x < self.W - 1:
                    basis = th.zeros((H, W))
                    basis[y, x] = 1
                    basis[y, x + 1] = 1
                    bases.append(basis)
                if y < self.H - 1:
                    basis = th.zeros((H, W))
                    basis[y, x] = 1
                    basis[y + 1, x] = 1
                    bases.append(basis)
        self.bases = th.stack(bases)
        if flatten:
            self.bases = self.bases.reshape((self.n_dict, -1))
        else:
            self.bases = self.bases
        return self.bases

class LTileLoader(BarsLoader):
    def __init__(self, H, W, n_batch, p=0.1, positive=True, test=False, numpy=False, sigma=1, l1=1):
        super().__init__(H, W, n_batch, p, positive, test, numpy, sigma, l1)
        self.set_bases()
    def set_bases(self, flatten=True):
        #bases = th.zeros((self.n_dict, self.H, self.W))
        bases = []
        for x in range(self.W - 1):
            for y in range(self.H - 1):
                square = th.zeros((self.H, self.W))
                square[x, y] = 1
                square[x + 1, y] = 1
                square[x, y + 1] = 1
                square[x + 1, y + 1] = 1
                for dx in (0, 1):
                    for dy in (0, 1):
                        basis = square.clone()
                        basis[x + dx, y + dy] = 0
                        bases.append(basis)
        self.bases = th.stack(bases)
        if flatten:
            self.bases = self.bases.reshape((self.n_dict, -1))
        else:
            self.bases = self.bases
        return self.bases

def paint_tile(tile_shape, x, y, coords):
    basis = th.zeros(tile_shape)
    for dy, dx in coords:
        basis[y + dy, x + dx] = 1
    return basis

class TTileLoader(BarsLoader):
    def __init__(self, H, W, n_batch, p=0.1, positive=True, test=False, numpy=False, sigma=1, l1=1):
        super().__init__(H, W, n_batch, p, positive, test, numpy, sigma, l1)
        self.set_bases()
    def set_bases(self, flatten=True):
        #bases = th.zeros((self.n_dict, self.H, self.W))
        bases = []
        for x in range(self.W - 1):
            for y in range(self.H - 1):
                if y < self.H - 2:
                    basis = paint_tile((self.H, self.W), x, y, [(0,0), (1,0), (2,0), (1,1)])
                    bases.append(basis)
                    basis = paint_tile((self.H, self.W), x, y, [(0,1), (1,1), (2,1), (1,0)])
                    bases.append(basis)
                if x < self.W - 2:
                    basis = paint_tile((self.H, self.W), x, y, [(0,0), (0,1), (0,2), (1,1)])
                    bases.append(basis)
                    basis = paint_tile((self.H, self.W), x, y, [(1,0), (1,1), (1,2), (0,1)])
                    bases.append(basis)


        self.bases = th.stack(bases)
        if flatten:
            self.bases = self.bases.reshape((self.n_dict, -1))
        else:
            self.bases = self.bases
        return self.bases

class SparseSampler():
    def __init__(self, A, n_batch, pi=0.2, l1=.2, sigma=0, positive=True, coeff='exp'):
        self.A = A
        self.n_dim, self.n_dict = A.shape
       
        self.n_batch = n_batch
        self.pi = pi
        self.l1 = l1
        self.sigma = sigma
        self.positive = positive
        self.coeff = coeff
    def get_coeff(self):
        if self.coeff == 'exp':
            s = th.FloatTensor(self.n_dict, self.n_batch).exponential_(self.l1)
            if not self.positive:
                s *= (1 - 2 * th.FloatTensor(*s.shape).bernoulli_(0.5))
            s *= th.FloatTensor(*s.shape).bernoulli_(self.pi)
        elif self.coeff == 'fixed':
            s = th.zeros(self.n_dict, self.n_batch)
            rand_idx = np.random.randint(self.n_dict, size=self.n_batch)
            for n in range(self.n_batch):
                s[rand_idx[n], n] = self.l1**(-1)
        else:
            print(self.coeff)
            raise Exception(f'Unknown coefficent sampling type')
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
    def __call__(self, n_batch=None, transposed=True):
        #TODO: implement n_batch
        return self.get_batch(transposed)

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

class VanHaterenSampler():
    def __init__(self, H, W, n_batch, flatten=True, torch=True, buffer_size=0):
        #self.img_ds = h5py.File('./vanhateren_imc/images.h5', 'a')[str(ds_size)]
        self.img_ds = h5py.File('./vanhateren_imc/images_bao.h5', 'r')['images']
        self.H = H
        self.W = W
        self.n_batch = n_batch
        self.flatten = flatten 
        self.torch = torch
        self.buffer = int(buffer_size)
        if self.buffer > 0:
            assert self.buffer > n_batch
            self.reset_buffer()
    def reset_buffer(self):
        self.buff_batch = self.sample(flatten=self.flatten, n_batch=self.buffer)
        self.buff_idx = 0
    def sample_buffer(self, n_batch):
        end_idx = self.buff_idx + n_batch
        if end_idx > self.buffer:
            self.reset_buffer
        buff = self.buff_batch[self.buff_idx: end_idx]
        self.buff_idx += n_batch
        return buff
    def sample(self, n_batch=None, flatten=None):
        if n_batch is None:
            n_batch = self.n_batch
        if flatten is None:
            flatten = self.flatten
        ds_size, i_max, j_max = self.img_ds.shape
        i_max -= self.W
        j_max -= self.H
        rand_n = np.random.randint(ds_size, size=n_batch)
        rand_i = np.random.randint(i_max, size=n_batch)
        rand_j = np.random.randint(j_max, size=n_batch)

        sample_arr = np.zeros((n_batch, self.W, self.H))
        for k in range(n_batch):
            n = rand_n[k]
            i = rand_i[k]
            j = rand_j[k]
            sample_arr[k] = self.img_ds[n, i:i+self.W, j:j+self.H]

        if flatten:
            sample_arr = sample_arr.reshape((n_batch, -1))
        if self.torch:
            sample_arr = th.tensor(sample_arr).float()
        return sample_arr
    def __call__(self, n_batch=None):
        if n_batch is None:
            n_batch = self.n_batch
        if self.buffer == 0:
            return self.sample(flatten=self.flatten, n_batch=n_batch)
        else:
            return self.sample_buffer(n_batch)

class GaussainSampler():
    def __init__(self, H, W, n_batch, sigma=1, buffer_size=0, flatten=True, torch=True):
        self.sigma = sigma
        self.H = H
        self.W = W
        self.n_batch = n_batch
        self.flatten = flatten 
        self.torch = torch
        self.buffer = int(buffer_size)
        if self.buffer > 0:
            assert self.buffer > n_batch
            self.reset_buffer()
    def reset_buffer(self):
        self.buff_batch = self.sample(flatten=self.flatten, n_batch=self.buffer)
        self.buff_idx = 0
    def sample_buffer(self, n_batch):
        end_idx = self.buff_idx + n_batch
        if end_idx > self.buffer:
            self.reset_buffer
        buff = self.buff_batch[self.buff_idx: end_idx]
        self.buff_idx += n_batch
        return buff
    def sample(self, n_batch=None, flatten=None):
        if n_batch is None:
            n_batch = self.n_batch
        if flatten is None:
            flatten = self.flatten

        sample_arr = np.random.normal(0, scale=self.sigma, size=(n_batch, self.W, self.H))

        if flatten:
            sample_arr = sample_arr.reshape((n_batch, -1))
        if self.torch:
            sample_arr = th.tensor(sample_arr).float()
        return sample_arr
    def __call__(self, n_batch=None):
        if n_batch is None:
            n_batch = self.n_batch
        if self.buffer == 0:
            return self.sample(flatten=self.flatten, n_batch=n_batch)
        else:
            return self.sample_buffer(n_batch)

if __name__ == '__main__':

    H = W = 8
    n_batch = 64
    vh_sampler = VanHaterenSampler(H, W, n_batch, buffer_size=1e3)

    X = vh_sampler()
    plt.figure(figsize=(8, 8))
    plot_dict(X.T, (8, 8), 8, 8, sort=False)
    plt.tight_layout()
    plt.savefig('./figures/vh_data.png')
