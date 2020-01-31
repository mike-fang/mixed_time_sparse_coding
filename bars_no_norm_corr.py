import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img, plot_dict
from plt_env import *
from soln_analysis import SolnAnalysis
from glob import  glob

lsc_path = f'results/bars_lsc/no_norm_A'
dirs = glob('./results/bars_lsc_oc_2/pi_*')
dirs.sort()
for d in dirs:
    print(d)

    analysis = SolnAnalysis(d)
    analysis.plot_nz_hist(start=.9)
    plt.show()
    if False:
        c = analysis.corr_mat_s(start=0.9)
        print(analysis.det_corr(c))
        plt.imshow(c)
        plt.show()
    #plot_lsc_dict()
#plot_lsc_A_norm(True)

