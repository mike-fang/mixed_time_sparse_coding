import torch as th
import numpy as np
from math import pi
import torch.nn.functional as F
import h5py
from scipy.io import loadmat
import matplotlib.pylab as plt
from time import time

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

class BarsLoader:
    def __init__(self, H, W, n_batch, p=0.1, positive=False, test=False):
        self.H = H
        self.W = W
        self.im_shape = (H, W)
        self.n_batch = n_batch
        self.p = p
        self.positive = positive
        self.test = test

        self.set_bases()
    def reset(self):
        pass
    def get_batch(self, reshape=False, n_batch=None, test=False):
        if self.test:
            test = True
        if n_batch is None:
            n_batch = self.n_batch
        S = th.Tensor(n_batch, self.W + self.H).bernoulli_(self.p)
        multiplier = th.Tensor(n_batch, self.W + self.H).uniform_()
        if self.positive:
            multiplier -= 0.5
        S *= multiplier
        if test:
            d_min = min(n_batch, self.W + self.H)
            S *= 0
            S[:d_min, :d_min] = th.eye(d_min)

        batch = S @ self.bases


        if reshape:
            return batch.reshape((n_batch, self.H, self.W))
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
    def __call__(self, n_batch=None):
        return self.get_batch(n_batch=n_batch)
    def __repr__(self):
        desc = 'HVLinesLoader\n'
        desc += f'H, W: {self.H}, {self.W}\n'
        desc += f'n_batch: {self.n_batch}\n'
        desc += f'p: {self.p}'
        return desc

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

if __name__ == '__main__':
    H = W = 16
    n_batch = 16
    vh_sampler = VanHaterenSampler(H, W, n_batch, buffer_size=1e3)
    t0 = time()
    ims = vh_sampler(16).reshape((16, H, W))
    print(time() - t0)
    for n, im in enumerate(ims):
        plt.subplot(4, 4, n+1)
        plt.imshow(im, cmap='Greys_r')
    plt.show()
