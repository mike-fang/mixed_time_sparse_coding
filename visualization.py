import matplotlib.pylab as plt
import numpy as np
from matplotlib import animation
from matplotlib import gridspec
import h5py
#from helpers import Solutions_H5

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
    fig, axes = plt.subplots(n_row, n_col)
    img_plot = []

    # Populate subplots with img plots
    for ax_row in axes:
        for ax in ax_row:
            img_plot.append(
                    ax.imshow([[0]], cmap='greys')
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
        print(f'Saving animation to {out_file}..')
        anim.save(out_file)
    else:
        plt.show()

def show_img_XRA(X, R, A, img_shape=None, n_frames=None, ratio=1.5, out_file=None):
    """
        params: The parameter to visualize, should be reshaped to be (n_frame_total, n_sparse_total, n_dim1, n_dim2).
        n_frames: Number of frames to be shown in animation, must be smaller than n_frame_total.
        n_comps: Number of sparse components to display msut be smaller than n_sparse_total.
        ratio: The aspect ratio to arrange axes for display.
    """

    if img_shape is not None:
        H, W = img_shape
        n_frames_total, n_batch, n_dim = X.shape
        n_frames_total, n_dim, n_dict = A.shape
        X = X.reshape((n_frames_total, n_batch, H, W))
        R = R.reshape((n_frames_total, n_batch, H, W))
        A = np.transpose(A, (0, 2, 1))
        A = A.reshape((n_frames_total, n_dict, H, W))

    n_frames_total, n_X, W, H = X.shape
    n_frames_total_R, n_R, W_R, H_R = R.shape
    n_frames_total_A, n_A, W_A, H_A = A.shape

    assert n_frames_total == n_frames_total_R == n_frames_total_A
    assert W == W_R == W_A
    assert H == H_R == H_A

    n_total = n_X + n_R + n_A

    if n_frames is None:
        skip = 1
        n_frames = n_frames_total
    else:
        n_frames = int(n_frames)
        skip = max(1, n_frames_total // n_frames)



    # Get n_rows / n_cols
    n_cols = ((n_total * ratio)**0.5)
    rows_X = int(np.ceil(n_X / n_cols))
    rows_R = int(np.ceil(n_R / n_cols))
    rows_A = int(np.ceil(n_A / n_cols))
    rows_total = rows_X + rows_A + rows_R
    n_cols = int(np.ceil(n_cols))

    # Make subplots
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(rows_total, n_cols, hspace=.3)
    gs_X = fig.add_subplot(gs[:rows_X, :])
    gs_R = fig.add_subplot(gs[rows_X:rows_X+rows_R, :])
    gs_A = fig.add_subplot(gs[rows_X+rows_R:rows_X+rows_R+rows_A, :])

    def make_img_gs(gs, n_rows, n_imgs, name):
        gs.set_xticks([])
        gs.set_yticks([])
        gs.set_title(name)
        inner_gs = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols,subplot_spec=gs, wspace=.05, hspace=.05)
        img_plots = []
        for n in range(n_imgs):
            ax = plt.Subplot(fig, inner_gs[n])
            img_plots.append(ax.imshow([[0]], cmap='Greys_r'))
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
        return img_plots

    plots_X = make_img_gs(gs_X, rows_X, n_X, 'Data')
    plots_R = make_img_gs(gs_R, rows_R, n_R, 'Reconstruction')
    plots_A = make_img_gs(gs_A, rows_A, n_A, 'Dictionary')

    def update_im(imgs, n_imgs, img_plots, frame_n):
        im = imgs[frame_n * skip]
        v_max = im.max()
        v_min = im.min()
        for i in range(n_imgs):
            img_plots[i].set_clim(v_min, v_max)
            img_plots[i].set_data(im[i])
            #img_plots[i].autoscale()

    def animate(n):
        update_im(X, n_X, plots_X, n)
        update_im(R, n_R, plots_R, n)
        update_im(A, n_A, plots_A, n)
        fig.suptitle(n)
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=True)
    #plt.tight_layout()
    if out_file is not None:
        print(f'Saving animation to {out_file}..')
        anim.save(out_file)
    else:
        plt.show()

def show_2d_evo(soln, n_frames=100, overlap=3, f_out=None, show_xmodel=False, show_smodel=False):
    X = soln['x']
    S = soln['s']
    t_steps, n_batch, n_dim = X.shape
    assert n_dim == 2
    _, n_dict, _ = S.shape
    idx_stride = int(t_steps / n_frames)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Create scatter plots for s and x
    sx, sy = [], []
    xx, xy = [], []
    scat_r_data = ax.scatter(sx, sy, s=5, c='b', label=rf'$A \mathbf {{u}}$ : Data Reconstruction')
    scat_x_data = ax.scatter(xx, xy, s=50, c='r', label=rf'$\mathbf {{x}}$ : Data')

    scat_r_model = ax.scatter(sx, sy, s=5, c='b', label=rf'$A \mathbf {{u}}$ : Model Reconstruction') if show_smodel else None
    scat_x_model = ax.scatter(xx, xy, s=50, c='b', label=rf'$\mathbf {{x}}$ : Model') if show_xmodel else None

    # Create arrows for tracking A
    A_arrow_0, = ax.plot([], [], c='g', label=rf'$A$ : Dictionary bases')
    A_arrows = [A_arrow_0]
    for A in range(n_dict):
        A_arrow, = ax.plot([], [], c='g')
        A_arrows.append(A_arrow)

    rng = 15
    ax.set_xlim(-rng, rng)
    ax.set_ylim(-rng, rng)
    ax.set_aspect(1)
    fig.legend(loc='lower right')

    def animate(nf):
        idx0 = max(0, (nf - overlap + 1) * idx_stride)
        idx1 = (nf + 1) * idx_stride 

        T = soln['t'][idx0:idx1]
        ti = T[0]
        tf = T[-1] 
        A = soln['A'][idx1]
        #r_model = soln['r_model'][idx0:idx1]
        r_data = soln['r'][idx0:idx1]
        #x_model = soln['x_model'][idx0:idx1].reshape((-1, n_dim))
        x_data = soln['x'][idx0:idx1].reshape((-1, n_dim))
    
        if scat_r_model:
            scat_r_model.set_offsets(r_model.reshape((-1, 2)))
            scat_r_model.set_array(np.linspace(0, 1, len(T)))
            scat_r_model.cmap = plt.cm.get_cmap('Blues')


        scat_r_data.set_offsets(r_data.reshape((-1, 2)))
        scat_r_data.set_array(np.linspace(0, 1, len(T)))
        scat_r_data.cmap = plt.cm.get_cmap('Blues')

        scat_x_data.set_offsets(x_data)
        if scat_x_model:
            scat_x_model.set_offsets(x_model)

        for n in range(n_dict):
            A_arrows[n].set_xdata([0, A[0, n]*5])
            A_arrows[n].set_ydata([0, A[1, n]*5])

        fig.suptitle(rf'Time: ${ti:.2f} \tau - {tf:.2f} \tau$')

    anim = animation.FuncAnimation(fig, animate, frames=n_frames-1, interval=100, repeat=True)
    if f_out is not None:
        anim.save(f_out)
    plt.show()

def show_2d_evo_no_model(soln, n_frames=100, overlap=3, f_out=None):
    X = soln['x']
    S = soln['s']
    t_steps, n_batch, n_dim = X.shape
    assert n_dim == 2
    _, n_dict, _ = S.shape
    idx_stride = int(t_steps / n_frames)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Create scatter plots for s and x
    sx, sy = [], []
    xx, xy = [], []
    scat_s = ax.scatter(sx, sy, s=5, c='b', label=rf'$A \mathbf {{u}}$ : Reconstruction')
    scat_x = ax.scatter(xx, xy, s=50, c='r', label=rf'$\mathbf {{x}}$ : Data')

    # Create arrows for tracking A
    A_arrow_0, = ax.plot([], [], c='g', label=rf'$A$ : Dictionary bases')
    A_arrows = [A_arrow_0]
    for A in range(n_dict):
        A_arrow, = ax.plot([], [], c='g')
        A_arrows.append(A_arrow)

    rng = 15
    ax.set_xlim(-rng, rng)
    ax.set_ylim(-rng, rng)
    ax.set_aspect(1)
    fig.legend(loc='lower right')

    def animate(nf):
        idx0 = max(0, (nf - overlap + 1) * idx_stride)
        idx1 = (nf + 1) * idx_stride 

        T = soln['t'][idx0:idx1]
        r = soln['r'][idx0:idx1]
        s = soln['s'][idx0:idx1].reshape((-1, n_dict))

        ti = T[0]
        tf = T[-1] 

        A = soln['A'][idx1]
    
        scat_s.set_offsets(r.reshape((-1, 2)))
        scat_s.set_array(np.linspace(0, 1, len(T)))
        scat_s.cmap = plt.cm.get_cmap('Blues')

        x = soln['x'][idx0:idx1].reshape((-1, n_dim))
        scat_x.set_offsets(x)

        for n in range(n_dict):
            A_arrows[n].set_xdata([0, A[0, n]])
            A_arrows[n].set_ydata([0, A[1, n]])

        fig.suptitle(rf'Time: ${ti:.2f} \tau - {tf:.2f} \tau$')

    anim = animation.FuncAnimation(fig, animate, frames=n_frames-1, interval=100, repeat=True)
    if f_out is not None:
        anim.save(f_out)
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
