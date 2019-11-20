import numpy as np
import matplotlib.pylab as plt
from matplotlib import animation
from time import time
from helpers import Loader, Solutions_H5, Solutions, save_model, load_model
import h5py
import os.path
from time import time
from tqdm import tqdm
import pickle
import shutil
from glob import glob

class Energy_L0:
    def __init__(self, positive=False):
        self.positive = positive
    def __call__(self, variables):
        A = variables['A']
        s = variables['s']
        x = variables['x']
        tau = variables['tau']
        l1 = variables['l1']
        s0 = variables['s0']

