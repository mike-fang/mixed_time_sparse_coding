import torch as th
import numpy as np
from ctsc import *
from loaders import BarsLoader
from visualization import show_img_XRA, show_batch_img, plot_dict
from plt_env import *
from soln_analysis import SolnAnalysis
from glob import  glob
from matplotlib import animation

def plot_lsc_A_norm(out=False):
    analysis.plot_dict_norm()
    plt.ylabel(r'Dict. Element Norm')
    plt.xlabel('Time')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.tight_layout()
    if out:
        plt.savefig('./figures/lsc_A_norm.pdf', bb_inches='tight')
    plt.show()

def plot_dict(A):
    A = A.reshape((8, 8, -1)).T
    A = A[order]

    A_join = np.ones((9*4 - 1, 9*8 -1 )) * 0.1
    for n, a in enumerate(A):
        i = n // 8
        j = n % 8
        A_join[i*9:i*9+8, j*9:j*9+8] = a

    plt.imshow(A_join, cmap='gray')
    plt.xticks([])
    plt.yticks([])

exp = 'lca'

if exp == 'lsc':
    path = get_timestamped_dir(load=True, base_dir='bars_lsc')
    analysis = SolnAnalysis(path)
elif exp == 'lca':
    path = get_timestamped_dir(load=True, base_dir='bars_lca', dir_name='no_norm_A')
    path = get_timestamped_dir(load=True, base_dir='bars_lca', dir_name=None)
    analysis = SolnAnalysis(path)

A = analysis.A
A_norm = np.linalg.norm(A, axis=1).T
order = (- A_norm[:, -1]).argsort()
time = analysis.time
NT = analysis.n_t



N_FRAMES = 100
skip = int(NT//N_FRAMES)

fig = plt.figure(figsize=(6, 8))

def animate(n):
    print(n)
    plt.clf()
    plt.cla()
    plt.subplot(211)
    for a in A_norm:
        plt.plot(time[:n*skip], a[:n*skip], 'k', lw=.5)
    plt.xlim(0, time.max())
    #plt.ylim(0, 1.5)
    plt.xlabel('Time')
    plt.ylabel('Dict. Element Norm')

    plt.subplot(212)
    plot_dict(A[n*skip])


anim = animation.FuncAnimation(fig, animate, frames=N_FRAMES, interval = 100, repeat=True )
#plt.show()
anim.save(f'./figures/bars_{exp}_oc_2_norm.mp4')
