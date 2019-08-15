import matplotlib.pylab as plt
import numpy as np
from matplotlib import animation
import h5py
from helpers import Solutions_H5

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

def show_img_evo(params, n_frames=None, n_comps=None, ratio=1.5, out_file=None):
    """
        params: The parameter to visualize, should be reshaped to be (n_frame_total, n_sparse_total, n_dim1, n_dim2).
        n_frames: Number of frames to be shown in animation, must be smaller than n_frame_total.
        n_comps: Number of sparse components to display msut be smaller than n_sparse_total.
        ratio: The aspect ratio to arrange axes for display.
    """
    frames_total, n_comps_total, n_dim1, n_dim2 = params.shape

    if n_frames is None:
        skip = 1
        n_frames = frames_total
    else:
        skip = frames_total // n_frames

    # Use all components if not specified
    if n_comps is None:
        n_comps = n_comps_total

    # Get n_rows / n_cols
    n_row = ((n_comps/ratio)**0.5)
    n_col = int(np.ceil(n_comps / n_row))
    n_row = int(np.ceil(n_row))

    # Make subplots
    fig, axes = plt.subplots(n_row, n_col, figsize=(15, 9))
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
    plt.tight_layout()
    if out_file is not None:
        anim.save(out_file)
        #animation.ImageMagickFileWriter()
    plt.show()

if __name__ == '__main__':
    soln = Solutions_H5('./results/hv_dsc_4x4/soln.h5')
    reshaped_params = soln.get_reshaped_params()
    A = reshaped_params['A']
    R = reshaped_params['R']
    X = reshaped_params['X']
    S = soln['S']

    sparsity = (S > 0).sum(axis=2)

    time_bins = 50
    n_dict = 8
    sparsity = sparsity[-100:].flatten()

    plt.figure()
    count = []

    plt.hist(sparsity + 0.5, bins=np.arange(11), density=True, ec='k', fc=(0.8,)*3)
    mu = sparsity.mean()/n_dict
    binom_pmf = binom.pmf(np.arange(n_dict + 1), n=n_dict, p=mu)
    plt.plot(np.arange(n_dict + 1) + 0.5, binom_pmf, 'ro-')
    plt.title(fr'$p \approx {mu:3f}$')
    plt.show()
