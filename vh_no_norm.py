import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img, plot_dict, animate_dict
from plt_env import *
from soln_analysis import SolnAnalysis
from matplotlib import animation

DIM = 8
def plot_dicts(out=False, sort=True):
    A = analysis.A[-1]
    norm = -np.linalg.norm(A, axis=0)
    plot_dict(A[:, norm.argsort()], (DIM, DIM), 8, 8)
    print(A.shape)
    #plt.suptitle('Sorted Learned Dictionary (128/128)')
    if out:
        plt.savefig('./figures/vh_unnormed_dict.pdf', bb_inches='tight')
    plt.show()
def plot_pi(out=False, pi=None):
    plt.figure(figsize = (8,4))
    print(analysis.pi)
    plt.plot(analysis.pi, 'k')
    if pi is not None:
        plt.plot(analysis.time, [pi,] * len(analysis.time), 'r--')
    plt.ylabel(r'Activity ($\pi$)')
    plt.xlabel('Time')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.tight_layout()
    if out:
        plt.savefig('./figures/lsc_pi', bb_inches='tight')
    plt.show()


#dir_path = f'results/vh_oc_2_dim_8_lsc/no_norm_A'
#base_dir = f'vh_learn_pi_dim_8_oc_2_ease_pi/'
base_dir = f'vh_learn_pi_dim_8_oc_2_fixed_pi/'
dir_path = get_timestamped_dir(load=True, base_dir=base_dir)
analysis = SolnAnalysis(dir_path)

if False:
    soln = analysis.soln
    X = soln['x'][:]
    R = soln['r'][:]
    A = soln['A'][:]

    show_img_XRA(X, R, A, n_frames=1e2, img_shape=(DIM, DIM), out_file=None)
    assert False

if True:
    A = analysis.A
    animate_dict(A, 100)


#dir_path = get_timestamped_dir(load=True, base_dir='vh_learn_pi_dim_16_oc_2')
#analysis = SolnAnalysis(dir_path)

plot_dicts(out=False)
plot_pi(False)

norm_A = np.linalg.norm(analysis.A, axis=1)
for norm in norm_A.T:
    plt.plot(analysis.time, norm, c='k', alpha=.9, lw=.3)
plt.tight_layout()
#plt.savefig('./figures/norm_evo.pdf', bb_inches='tight')
plt.show()

