from euler_maruyama import EulerMaruyama
import torch as th
import numpy as np
from torch.nn import Parameter
from tqdm import tqdm
from plt_env import *

u = Parameter(th.tensor(0.))


def energy():
    E = th.abs(u)
    E.backward()
    return E

steps = int(1e5)
param_groups = [
        {'params': [u], 'tau': 5e+1, 'T': 1.000},
        ]
solver = EulerMaruyama(param_groups)

try:
    U = np.load('./results/u.npy')
except:
    U = np.zeros(steps)
    for n in tqdm(range(steps)):
        solver.zero_grad()
        solver.step(energy)
        U[n] = np.abs(u.data.numpy())
    np.save('./results/u.npy', U)

DIAG_LENGTH = .05

def plot_U():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, facecolor='w', figsize=(8, 4))
    s_max = 5
    trange = np.arange(steps) / steps
    ax1.plot(trange, U, 'k')
    ax1.plot((0, 1), (1, 1), 'b-.')
    ax2.plot(trange, U, 'k', label=r'$u(t)$')
    ax2.plot((0, 1), (1, 1), 'b-.', label=r'$u_0$')
    ax2.legend()

    ax1.set_xlim(0, 0.01)
    ax2.set_xlim(.01, 1)

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    #ax1.yaxis.tick_left()
    #ax1.tick_params(labelright='off')
    ax1.set_ylim(0, s_max)
    ax2.set_ylim(0, s_max)
    ax3.set_ylim(0, s_max)
    fig.text(0.33, 0.03, 'Time(a.u.)', fontsize=15)
    ax2.set_yticks([])
    #ax1.set_ylabel(r'u(t)')
    ax3.set_xlabel(r'p(u)')

    #ax1.set_yticks(np.linspace(0, s_max, 5))

    d = DIAG_LENGTH
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)

    s_list = np.linspace(0, s_max, 100)

    ax3.hist(U, bins=np.linspace(0.01, s_max, 20), fc='grey', orientation='horizontal', density=True, label='Emp. Distr.')
    ax3.plot(np.exp(-s_list), s_list, 'r--', label=r'$\lambda e^{-\lambda u}$')
    plt.tight_layout()
    fig.subplots_adjust(top=.85, bottom=.15)
    fig.suptitle(r'$\lambda = 1; u_0 = 1; \pi = e^{-\lambda u_0} \approx 0.368 $', fontsize=16)

    plt.legend()

def plot_S():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, facecolor='w', figsize=(8, 4))
    S = U - 1
    S[S < 0] = 0
    s_max = 5
    trange = np.arange(steps) / steps
    ax1.plot(trange, S, 'k')
    ax2.plot(trange, S, 'k', label=r'$s(t)$')
    ax2.legend()

    ax1.set_xlim(0, 0.01)
    ax2.set_xlim(.01, 1)

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    #ax1.yaxis.tick_left()
    #ax1.tick_params(labelright='off')
    ax1.set_ylim(0, s_max)
    ax2.set_ylim(0, s_max)
    ax3.set_ylim(0, s_max)
    fig.text(0.33, 0.03, 'Time(a.u.)', fontsize=15)
    ax2.set_yticks([])
    #ax1.set_ylabel(r'u(t)')
    ax3.set_xlabel(fr'$p(s(t) = s | s > 0)$')

    #ax1.set_yticks(np.linspace(0, s_max, 5))

    d = DIAG_LENGTH
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)

    s_list = np.linspace(0, s_max, 100)

    ax3.hist(S[S > 0], bins=np.linspace(0.00, s_max, 20), fc='grey', orientation='horizontal', density=True, label='Emp. Distr.')
    ax3.plot(np.exp(-s_list), s_list, 'r--', label=r'$\lambda e^{-\lambda u}$')
    plt.tight_layout()
    fig.subplots_adjust(top=.85, bottom=.15)
    fig.suptitle(rf'$p(s>0) = {(S > 0).mean():.3f} $', fontsize=16)
    plt.legend()

plot_U()
plt.savefig('./figures/p_u.pdf', bb_inches='tight')
plt.show()
plot_S()
plt.savefig('./figures/p_s.pdf', bb_inches='tight')
plt.show()


