import torch as th
import sys
sys.path.append('..')
from ctsc import CTSCModel, CTSCSolver, get_timestamped_dir
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img, plot_dict
from plt_env import *
from soln_analysis import SolnAnalysis

DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'
