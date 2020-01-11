from soln_analysis import *
import matplotlib.pyplot as plt
from matplotlib import gridspec

def remove_border(ax):
    for pos in ['top', 'bottom', 'left', 'right']:
        ax.spines[pos].set_visible(False)
def populate_sgs(sub_gs, data, symbol, title, t_labels=False):
    remove_border(sub_gs)
    sub_gs.set_title(title)
    sub_gs.set_xticks([])
    sub_gs.set_yticks([])
    inner_gs = gridspec.GridSpecFromSubplotSpec(4, N_COL, subplot_spec=sub_gs, wspace=.05, hspace=.05)

    for n in range(N_COL):
        for i in [0, 1, -1]:
            ax = plt.Subplot(fig, inner_gs[i, n])
            ax.imshow(data[int(tau_x * n / t_frames), i])
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            if n == 0:
                if i == -1:
                    i = 'N'
                ax.set_ylabel(fr'${symbol}_{i}$')

    ax = plt.Subplot(fig, inner_gs[2, 0])
    ax.set_xticks([])
    ax.set_yticks([])
    remove_border(ax)
    ax.set_ylabel(r'$\dots$')
    fig.add_subplot(ax)

ds_name = 'vh_dim_8_dsc'

fig = plt.figure(figsize=(12, 8))

t_mult = 5
t_frames = 3

N_COL = t_mult * t_frames

im_shape = (8,8)
tau_x = 100
X = np.random.randn(1000, 10, 8, 8)
T = np.arange(1000)

main_gs = fig.add_gridspec(3, N_COL)

gs_X = fig.add_subplot(main_gs[0, :])
gs_R = fig.add_subplot(main_gs[1, :])
gs_A = fig.add_subplot(main_gs[2, :])

populate_sgs(gs_X, X, 'x', 'Data')
populate_sgs(gs_R, X, r'\hat x', 'Reconstruction')
populate_sgs(gs_A, X, 'A', 'Dictionary')




plt.show()
