import pickle
import h5py
import numpy as np
from time import time

T = int(1e7)
S = 10
D = 25

#array = np.random.rand(T, S, D)

class Solutions:
    @classmethod
    def from_h5(cls, f_name):
        soln = cls()
        soln.h5 =h5py.File(f_name, 'a')
        return soln
    def __init__(self, a=None):
        if a is not None:
            self.h5 = h5py.File('./test_file.h5', 'a')
            self.h5.create_dataset('array', data=a)

t0 = time()
soln = Solutions.from_h5('./test_file.h5')
print(f'soln object loaded in {time() - t0:.2f}s')
print(soln.h5['array'][0])

assert False
print('Rand N generated')
t0 = time()
#soln = Solutions(array)
print(f'soln object made in {time() - t0:.2f}s')
t0 = time()
with open('./test_file.pkl', 'wb') as f:
    pickle.dump(soln, f)
print(f'picked in {time() - t0:.2f} s')
