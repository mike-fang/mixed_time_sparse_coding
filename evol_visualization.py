from soln_analysis import *
from visualization import show_evol
from matplotlib import gridspec

CX = 'k'
CR = 'g'
CA = 'm'
AW = .5


def plot_XR(X, R, T, tau_x, ax, n, d=None):
    if d is None:
        _, _, D = X.shape
        rand_proj = np.random.randn(D)
        X_proj = X[:, n, :] @ rand_proj
        R_proj = R[:, n, :] @ rand_proj
    else:
        X_proj = X[:, n, d]
        R_proj = R[:, n, d]

    ax.plot(T / tau_x, X_proj, C=CX, label=rf'$x_{n+1}$')
    ax.plot(T / tau_x, R_proj, c=CR, ls=':', label=rf'$s_1$')
    plt.ylabel('a.u.', fontsize=14)
    plt.yticks([])
    plt.xlabel('')
    plt.xticks([])
def plot_evo(X, R, A, T, tau_x):
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(5, 1, hspace=.5)

    ax = fig.add_subplot(gs[0, :])
    plt.title('Input and Coefficient', fontsize=18)
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
    plt.legend(loc=2)
def plot_evo_one(X, R, A, T, tau_x, exp):
    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(2, 1, hspace=.3)

    ax = fig.add_subplot(gs[0, :])
    plt.title('Input and Coefficients', fontsize=18)
    plot_XR(X, R, T, tau_x, ax, 0, d=0)
    
    if exp == 'lsc':
        X = 2
        Y = 1.1
        c=CX
        plt.annotate(s=r'$\tau_x$', fontsize=14, xy=(X+.5, Y + .2), xytext=(X+.5,Y+.2), c=c)
        plt.annotate(s='', xy=(X+1, Y), xytext=(X,Y), arrowprops=dict(arrowstyle=f'|-|, widthA={AW}, widthB={AW}', color=c))

        c=CR
        X = 2
        Y = -.7
        plt.annotate(s=r'$\tau_s$', fontsize=14, xy=(X+.2, Y), xytext=(X+.25,Y-.15), c=c)
        plt.annotate(s='', xy=(X+.55, Y), xytext=(X+.45,Y), arrowprops=dict(arrowstyle=f'|-|, widthA={AW}, widthB={AW}', color=c))


    y_lim = ax.get_ylim()

    if exp == 'dsc':
        for x in range(5):
            x1 = 0.97 + x
            x2 = 1 + x
            u = x/4
            c = (1-u, u * (1-u), u)
            ax.fill_betweenx(y_lim, [x1, x1], [x2, x2], fc=c, alpha=.5)
    elif exp == 'slsc':
        for x in range(5):
            x1 = 0.02 + x
            x2 = 1 + x
            u = x/4
            c = (1-u, u * (1-u), u)
            ax.fill_betweenx(y_lim, [x1, x1], [x2, x2], fc=c, alpha=.2)
    elif exp == 'lsc':
        x1 = 0
        for t in (T[::5] / tau_x):
            x2 = t
            u = t/5.4
            c = (1-u, u * (1-u), u)
            ax.fill_betweenx(y_lim, [x1, x1], [x2, x2], fc=c, alpha=.2)
            x1 = x2

    ax.fill_betweenx([], [], [], fc='grey', alpha=.5)
    plt.legend(loc=2)

    ax = fig.add_subplot(gs[1:, :])

    if exp == 'lsc':
        c=CA
        X = 2.5
        Y = -.3
        plt.annotate(s='', xy=(X-1.5, Y), xytext=(X+1.5,Y), arrowprops=dict(arrowstyle=f'|-|, widthA={AW}, widthB={AW}', color=c))
        plt.annotate(s=r'$\tau_A$', fontsize=14, xy=(X+.2, Y), xytext=(X,Y-.02), c=c)

    ax.plot(T / tau_x, A[:, 5, 2], c=CA, ls='-.', label=r'$A_1$')
    y_lim = ax.get_ylim()

    if exp != 'lsc':
        for x in range(5):
            x1 = 1 + x
            x2 = 1.03 + x
            u = x/4
            c = (1-u, u * (1-u), u)
            ax.fill_betweenx(y_lim, [x1, x1], [x2, x2], fc=c, alpha=.5)
    elif exp == 'lsc':
        x1 = 0
        for t in (T[::5] / tau_x):
            x2 = t
            u = t/5.4
            c = (1-u, u * (1-u), u)
            ax.fill_betweenx(y_lim, [x1, x1], [x2, x2], fc=c, alpha=.2)
            x1 = x2
    ax.fill_betweenx([], [], [], fc='grey', alpha=.5)
    plt.title('Dictionary', fontsize=18)
    plt.ylabel('a.u.', fontsize=14)
    plt.yticks([])
    plt.xlabel(r'Time ($\tau_x$)', fontsize=14)
    plt.legend(loc=2)
def plot_exp_one(exp, save=False):
    base_dir = f'bars_large_step_{exp}'
    dir_path = get_timestamped_dir(load=True, base_dir=base_dir)
    analysis = SolnAnalysis(dir_path)

    if exp == 'lsc':
        print(analysis.tau_x)
        print(analysis.tau_s)
        print(analysis.tau_A)

    X = analysis.soln['x']
    tau_x = analysis.tau_x
    N, _, _ = X.shape
    N_start = int(4 * tau_x)
    N_end = int(9.2 * tau_x)
    X = analysis.soln['x'][N_start:N_end]
    R = analysis.soln['r'][N_start:N_end]
    A = analysis.soln['A'][N_start:N_end]
    T = analysis.soln['t'][N_start:N_end]
    T -= T.min()
    im_shape = (8, 8)

    cts = not (exp == 'dsc')
    cta = (exp == 'lsc')

    plot_evo_one(X, R, A, T, tau_x, exp=exp)
    plt.gcf().subplots_adjust(left=0.01, right=.99, bottom=0.13)
    if save:
        plt.savefig(f'./figures/evol_vis_1d_{exp}.pdf', bbox_inches='tight')
    plt.show()

plot_exp_one('dsc', save=True)
plot_exp_one('slsc', save=True )
plot_exp_one('lsc', save=True)

assert False



t_mult = 5
t_frames = 4

show_evol(X, R, A, im_shape, tau_x, 5)

plt.savefig(f'./figures/evol_vis_{exp}.pdf', bbox_inches='tight')
#plt.show()
