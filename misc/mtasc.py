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
FILE_DIR = os.path.abspath(os.path.dirname(__file__))

class Energy_L0:
    def __init__(self, positive=False):
        self.positive = positive
