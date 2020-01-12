from soln_analysis import SolnAnalysis
import numpy as np
from ctsc import get_timestamped_dir
import matplotlib.pylab as plt
import seaborn as sns

EXP = 'dsc'
base_dir = f'bars_{EXP}'
dir_path = get_timestamped_dir(load=True, base_dir=base_dir)
analysis = SolnAnalysis(dir_path)

S_soln = analysis.soln['s'][:]
t_inter = 100
n_t = len(S_soln)
n_inter = int(n_t // t_inter)
