import numpy as np
import matplotlib.pylab as plt
import pickle
from glob import glob
import os.path
from time import time
import h5py
import shutil

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
def get_tmp_path(load=False, f_name=None):
    if load:
        tmp_files = glob(os.path.join(FILE_DIR, 'results', 'tmp', '*'))
        tmp_files.sort()
        dir_name = tmp_files[-1]
    else:
        time_stamp = f'{time():.0f}'
        dir_name = os.path.join(FILE_DIR, 'results', 'tmp', time_stamp)
        os.makedirs(dir_name)
    if f_name is None:
        return dir_name
    else:
        return os.path.join(dir_name, f_name)

class Solutions:
    @classmethod
    def load(cls, f_name=None):
        print(f_name)
        if f_name is None:
            # Pick the newest tmp file if none given
            f_name = get_tmp_path(load=True, f_name='soln.h5')
        print(f'Loading solutions from {f_name}')
        return cls(f_name=f_name)
        #self.h5_file = h5py.File(f_name, 'r')
    def __init__(self, solns=None, f_name=None, im_shape=None, overwrite=False):
        self.init_h5_file(f_name, overwrite)
        if solns is not None:
            self.h5_file.attrs['im_shape'] = im_shape or ()
            self.save_soln(solns)
        self.get_shape()
    def init_h5_file(self, f_name, overwrite):
        if f_name in [None, 'temp']:
            # Output to temp file if none specified
            f_name = get_tmp_path(f_name='soln.h5')
        self.f_name = f_name
        if os.path.isfile(f_name) and overwrite:
            os.unlink(f_name)
        self.h5_file = h5py.File(f_name, 'a')
    def save_soln(self, solns):
        #self.solns = solns
        for key, val in solns.items():
            print(key, val.shape)
            print(key, val.dtype)
            self.h5_file.create_dataset(key, data=val)
        if not 'r_data' in self.h5_file:
            S = solns['s_data']
            A = solns['A']
            R = np.einsum('ijk,ilk->ijl', S, A)
            self.h5_file.create_dataset('r_data', data=R)
        if not 'r_model' in self.h5_file:
            S = solns['s_model']
            A = solns['A']
            R = np.einsum('ijk,ilk->ijl', S, A)
            self.h5_file.create_modelset('r_model', data=R)
    def get_shape(self):
        self.n_frame, self.n_dim, self.n_dict = self.h5_file['A'].shape
        _, self.n_batch, _ = self.h5_file['x_data'].shape
    def get_reshaped_params(self, indices=None):
        try:
            im_shape = self.h5_file['im_shape']
        except:
            im_shape = self.h5_file.attrs['im_shape']
        if im_shape is 'None':
            print('No im_shape provided')
            return None
        else:
            H, W = im_shape

        if indices is None:
            A = np.transpose(self.h5_file['A'][:], (0, 2, 1))
            R  = self.h5_file['r_data'][:]
            X  = self.h5_file['x_data'][:]
            R_  = self.h5_file['r_model'][:]
            X_  = self.h5_file['x_model'][:]
        else:
            # Only retrieve needed params
            A = np.transpose(self.h5_file['A'][indices], (0, 2, 1))
            R  = self.h5_file['r_data'][indices]
            X  = self.h5_file['x_data'][indices]
            R_  = self.h5_file['r_model'][indices]
            X_  = self.h5_file['x_model'][indices]

        reshaped_params = {}
        reshaped_params['A'] = A.reshape((-1, self.n_dict, H, W))
        reshaped_params['r_data'] = R.reshape((-1, self.n_batch, H, W))
        reshaped_params['x_data'] = X.reshape((-1, self.n_batch, H, W))
        reshaped_params['r_model'] = R_.reshape((-1, self.n_batch, H, W))
        reshaped_params['x_model'] = X_.reshape((-1, self.n_batch, H, W))
        return reshaped_params
    def __getitem__(self, key):
        return self.h5_file[key][:]
    def __getattr__(self, key):
        return self.h5_file[key][:]
