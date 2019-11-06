from euler_maruyama import *
import torch as th
import numpy as np
import matplotlib.pylab as plt
from torch.nn import Parameter
from matplotlib import animation
from tqdm import tqdm

steps = int(2e5)


NEW = False
TX0 = 10
TX = 5
NAME = 'no_mass'
MODEL = 'cauchy'
GAMMA = 2.5
RANGE = 15
def get_dynamics(tau_x0, tau_x):
    x = Parameter(th.tensor(5.))
    x0 = Parameter(th.tensor(3.))
    tau_x0 = tau_x0

    param_groups = [
            {'params': [x], 'tau': tau_x, 'mu': 0, 'T': .25},
            ]

    X = th.zeros(steps)
    X0 = th.zeros(steps)

    def closure():
        #energy = 0.5 * (x - x0)**2
        energy = E_cauchy(x)
        energy.backward()
        return energy

    solver = EulerMaruyama(param_groups, dt=1)
    for n in tqdm(range(steps)):
        #x0.data = 3 * th.sin(th.tensor(n/tau_x0))
        if n % tau_x0 == 0:
            x0.data *= -1
        solver.zero_grad()
        solver.step(closure)
        X[n] = float(x)
        X0[n] = float(x0)
    return X, X0

def p_cauchy(x):
    return (np.pi * GAMMA * (1 + (x/GAMMA)**2))**(-1)
def E_cauchy(x):
    try:
        return th.log(1 + (x/GAMMA)**2)
    except:
        return np.log(1 + (x/GAMMA)**2)

try:
    X = np.load(f'./{MODEL}_{NAME}.npy')
    X0 = np.load(f'./{MODEL}_{NAME}_X0.npy')
    if NEW:
        assert False
except:
    X, X0 = get_dynamics(TX0, TX)
    np.save(f'./{MODEL}_{NAME}.npy', X)
    np.save(f'./{MODEL}_{NAME}_X0.npy', X0)

x = np.linspace(-RANGE, RANGE, 100)
p = 0.5 * np.exp(-0.5 * (x-3)**2) / (2 * np.pi)**0.5
p += 0.5 * np.exp(-0.5 * (x+3)**2) / (2 * np.pi)**0.5
p_mean = np.exp(-0.5 * (x)**2) / (2 * np.pi)**0.5
p_cauchy = p_cauchy(x)
p_sine =  (9 - x**2)**(-0.5)/np.pi
p_sine[np.isnan(p_sine)] = 0
p_sine = np.convolve(p_sine, p_mean, mode='same')
p_sine /= p_sine.sum() * (x[1]-x[0])

N_FRAMES = 200
N_SKIP = int(steps//N_FRAMES)
#fig, axes = plt.subplots(nrows=2)
#fig, ax = plt.subplots()
fig, axes = plt.subplots(nrows=3)
def animate_hist(nf):
    for ax in axes:
        ax.clear()
    N = nf * N_SKIP
    N0 = max(0, (nf-5) * N_SKIP)
    _X = X[:N]
    _X0 = X0[:N]
    axes[0].plot(range(N0, N), X[N0:N], 'k')
    axes[0].plot(range(N0, N), _X0[N0:N], 'g--')
    axes[0].plot([N0, N0], [-RANGE, RANGE], 'r:')
    axes[0].plot([N, N], [-RANGE, RANGE], 'r:')
    axes[0].plot([N, N0], [RANGE, RANGE], 'r:')
    axes[0].plot([N, N0], [-RANGE, -RANGE], 'r:')
    axes[1].plot(_X, 'k')
    axes[1].plot(_X0, 'g--', alpha=.5)
    axes[1].plot([N0, N0], [-RANGE, RANGE], 'r:')
    axes[1].plot([N, N], [-RANGE, RANGE], 'r:')
    axes[1].plot([N, N0], [RANGE, RANGE], 'r:')
    axes[1].plot([N, N0], [-RANGE, -RANGE], 'r:')
    axes[2].hist(_X, log=True, bins=x, fc=(.3,)*3, density=True)
    #axes[2].plot(x, p, 'c--')
    #axes[2].plot(x, p_mean, 'm--')
    axes[2].plot(x, p_cauchy, 'm--')
    #axes[2].plot(x, p_sine, 'm--')
    axes[0].set_ylim(-RANGE*1.1, RANGE*1.1)
    axes[1].set_xlim(0, steps)
    axes[1].set_ylim(-RANGE*1.1, RANGE*1.1 )
    axes[2].set_ylim(.001, .2)
    axes[2].set_yscale('log')

def animate_(nf):
    for ax in axes:
        ax.clear()
    axes[0].plot(x, E_cauchy(x), 'r')
    x_nf = X[nf]
    y_nf = np.interp(x_nf, x, E_cauchy(x))
    axes[0].plot(x_nf, y_nf, 'ko')
    axes[1].plot(X[:nf], range(0, nf), 'k')
    axes[0].set_xlim(-RANGE, RANGE)
    axes[1].set_xlim(-RANGE, RANGE)

def animate(nf):
    ax.clear()
    x0 = X0[nf*10]
    E = 0.5 * (x-x0)**2
    ax.plot(x, E, 'r')
    ax.set_ylim(-1, 15)
    ax.set_xlim(-RANGE, RANGE)
    x_nf = X[nf*10]
    y_nf = np.interp(x_nf, x, E)
    ax.plot(x_nf, y_nf, 'ko')

anim = animation.FuncAnimation(fig, animate_hist, frames=N_FRAMES-1, interval=100, repeat=True)

anim.save(f'./{MODEL}_{NAME}_evo.mp4')
plt.show()
