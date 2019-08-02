import matplotlib.pylab as plt
import numpy as np
from matplotlib import animation
import h5py
from loaders import Solutions

def get_grid_axes(n_axes, ratio=1.5):
    # Get n_rows / n_cols
    n_row = ((ratio * n_axes)**0.5)
    n_col = int(np.ceil(n_axes / n_row))
    n_row = int(np.ceil(n_row))

    # Make subplots
    fig, axes = plt.subplots(n_row, n_col)
    img_plot = []

    # Populate subplots with img plots
    for ax_row in axes:
        for ax in ax_row:
            img_plot.append(
                    ax.imshow([[0]])
                    )
            ax.set_xticks([])
            ax.set_yticks([])

def show_img_evo(params, n_frames, n_comps=None, ratio=1.5):
    """
        params: The parameter to visualize, should be reshaped to be (n_frame_total, n_sparse_total, n_dim1, n_dim2).
        n_frames: Number of frames to be shown in animation, must be smaller than n_frame_total.
        n_comps: Number of sparse components to display msut be smaller than n_sparse_total.
        ratio: The aspect ratio to arrange axes for display.
    """
    print(params.shape)
    frames_total, n_comps_total, n_dim1, n_dim2 = params.shape
    skip = frames_total // n_frames

    # Use all components if not specified
    if n_comps is None:
        n_comps = n_comps_total

    # Get n_rows / n_cols
    n_row = ((n_comps/ratio)**0.5)
    n_col = int(np.ceil(n_comps / n_row))
    n_row = int(np.ceil(n_row))

    # Make subplots
    fig, axes = plt.subplots(n_row, n_col)
    img_plot = []

    # Populate subplots with img plots
    for ax_row in axes:
        for ax in ax_row:
            img_plot.append(
                    ax.imshow([[0]])
                    )
            ax.set_xticks([])
            ax.set_yticks([])

    def animate(n):
        A = params[n * skip]
        for i in range(n_comps):
            a = A[i]
            img_plot[i].set_data(a)
            img_plot[i].autoscale()
        fig.suptitle(n)
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=True)
    plt.show()

if __name__ == '__main__':
    print('hi')
    #solns = h5py.File('./results/hv_line_HV_init.h5py', 'r')
    #solns = h5py.File('./results/hv_line_HV.h5py', 'r')
    solns = Solutions.load('./results/hv_line_mtsc.soln')
    #solns = Solutions.load('./results/hv_line_dsc.soln')
    H, W = 5, 5


    A = solns.A_reshaped
    X = solns.X_reshaped
    R = solns.R_reshaped

    XRA = np.concatenate((X, R, A), axis=1)
    show_img_evo(XRA, n_frames=100, ratio=(10/3))
