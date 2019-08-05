import numpy as np
from glob import glob
import h5py

class DictTest:
    def __init__(self, **dict_):
        self.dict = dict_
    def __repr__(self):
        string = "Energy_L0\n"
        for k, v in self.dict.items():
            string += f'{k}: {v}\n'

        return string

params = {
        'a' : 1,
        'b' : 2,
        'c' : 3,
        }

dict_test = DictTest(**params)
string = "test\n"
string += f'{dict_test}'
print(string)
