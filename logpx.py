import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img, plot_dict
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis


dir_path = get_timestamped_dir(load=True, base_dir='bars_lsc')
analysis = SolnAnalysis(dir_path)
As = analysis.A
NR = 10000
NX = 1000

loader = BarsLoader(8, 8, NX, p=0.3, sigma=0.0, l1=1, numpy=True)
for A in As[::10]:
    R_loader = BarsLoader(8, 8, NR, p=0.3, sigma=0.0, l1=1, numpy=True)
    R_loader.bases = th.tensor(A.T)
    X = loader()
    R = R_loader()
    err = ((X[None, :, :] - R[:, None, :])**2).sum(axis=2)
    px = err.mean(axis=0)
    nll = -np.log(px).mean()
    print(nll)

