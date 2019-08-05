import numpy as np
from glob import glob
import h5py

class DictTest:
    def __init__(self, **dict_):
        self.dict = dict_

params = {
        'a' : 1,
        'b' : 2,
        'c' : 3,
        }

dict_test = DictTest(**params)
print(dict_test.dict)

def update_param(x, p, dEdx, dt, tau, mu, T, dW):
    m = mu * tau**2
    x += p * dt / (2*m)
    p += -tau * p * dt / m - dEdx
    if T > 0:
        p += (T * tau)**0.5 * dW
    x += p * dt / (2*m)
    return x, p



