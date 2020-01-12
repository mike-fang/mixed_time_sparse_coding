from soln_analysis import *
from visualization import show_evol
from matplotlib import gridspec

exp = 'asynch'
base_dir = f'bars_large_step_{exp}'
dir_path = get_timestamped_dir(load=True, base_dir=base_dir)
analysis = SolnAnalysis(dir_path)

X = analysis.soln['x']
tau_x = analysis.tau_x
N, _, _ = X.shape

X = analysis.soln['x'][N//2:]
R = analysis.soln['r'][N//2:]
A = analysis.soln['A'][N//2:]
T = analysis.soln['t'][N//2:]
T -= T.min()
im_shape = (8, 8)


def plot_XR(ax, n, d=None):
    if d is None:
        _, _, D = X.shape
        rand_proj = np.random.randn(D)
        X_proj = X[:, n, :] @ rand_proj
        R_proj = R[:, n, :] @ rand_proj
    else:
        X_proj = X[:, n, d]
        R_proj = R[:, n, d]

    ax.plot(T / tau_x, X_proj, 'k', label=rf'$x_{n+1}$')
    ax.plot(T / tau_x, R_proj, 'k:', label=rf'$\hat x_{n+1}$')
    plt.ylabel('a.u.', fontsize=14)
    plt.yticks([])
    plt.xlabel('')
    plt.xticks([])
    plt.legend(loc=1)
def plot_evo(X, R, A, T, tau_x):
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(5, 1, hspace=.5)

    ax = fig.add_subplot(gs[0, :])
    plt.title('Input and Reconstruction', fontsize=18)
    plot_XR(ax, 0)
    ax = fig.add_subplot(gs[1, :])
    plot_XR(ax, 1)
    ax = fig.add_subplot(gs[2, :])
    plot_XR(ax, 2)

    ax = fig.add_subplot(gs[3:, :])
    ax.plot(T / tau_x, A[:, 0, 0], 'r', label=r'$A_1$')
    ax.plot(T / tau_x, A[:, 1, 0], 'g', label=r'$A_2$')
    ax.plot(T / tau_x, A[:, 2, 0], 'orange', label=r'$A_3$')
    ax.plot(T / tau_x, A[:, 3, 0], 'b', label=r'$A_4$')
    plt.title('Dictionary', fontsize=18)
    plt.ylabel('a.u.', fontsize=14)
    plt.yticks([])
    plt.xlabel(r'Time ($\tau_x$)', fontsize=14)
    plt.legend(loc=1)

plot_evo(X, R, A, T, tau_x)
plt.savefig(f'./figures/evol_vis_1d_{exp}.pdf', bbox_inches='tight')
plt.show()

assert False



t_mult = 5
t_frames = 4

show_evol(X, R, A, im_shape, tau_x, 5)

plt.savefig(f'./figures/evol_vis_{exp}.pdf', bbox_inches='tight')
#plt.show()
