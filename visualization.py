import matplotlib.pylab as plt
import numpy as np
from matplotlib import animation
from matplotlib import gridspec
import h5py
#from helpers import Solutions_H5

CMAP = 'Greys_r'

def show_batch_img(ims, im_shape=None, ratio=1.5):
    # Get n_rows / n_cols
    N = len(ims)
    n_row = ((ratio * N)**0.5)
    n_col = int(np.ceil(N / n_row))
    n_row = int(np.ceil(n_row))

    # Make subplots
    fig, axes = plt.subplots(n_row, n_col)
    axes = [ax for row in axes for ax in row]

    # Populate subplots with img plots
    n = 0
    for n, ax in enumerate(axes):
        try:
            im = ims[n]
            if im_shape is not None:
                im = im.reshape(im_shape)
        except:
            im = [[0]]

        ax.imshow(im, cmap=CMAP)
        ax.set_xticks([])
        ax.set_yticks([])
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
                    ax.imshow([[0]], cmap=CMAP)
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
def show_img_XRA(X, R, A, img_shape=None, n_frames=None, ratio=1.5, out_file=None, N_batch=None, N_dict=None):
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

    if N_batch is not None:
        X = X[:, :N_batch, :, :]
        R = R[:, :N_batch, :, :]
    if N_dict is not None:
        A = A[:, :N_dict, :, :]

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
            img_plots.append(ax.imshow([[0]], cmap=CMAP))
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
def show_evol(X, R, A, im_shape, tau_x, t_mult, n_frames=4):
    def remove_border(ax):
        for pos in ['top', 'bottom', 'left', 'right']:
            ax.spines[pos].set_visible(False)
    def populate_sgs(sub_gs, data, symbol, title, plot_A=False):
        remove_border(sub_gs)
        sub_gs.set_title(title)
        sub_gs.set_xticks([])
        sub_gs.set_yticks([])
        inner_gs = gridspec.GridSpecFromSubplotSpec(4, n_cols, subplot_spec=sub_gs, wspace=.05, hspace=.05)

        for n in range(n_cols):
            for i in [0, 1, -1]:
                tau_frac = n / n_frames
                idx = int(tau_frac * tau_x)
                if plot_A:
                    im = data[idx, :, i]
                else:
                    im = data[idx, i]
                im = im.reshape(im_shape)

                ax = plt.Subplot(fig, inner_gs[i, n])
                ax.imshow(im, cmap=CMAP)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)

                if plot_A and (i == -1):
                    if tau_frac % .5 == 0:
                        ax.set_xlabel(fr'${tau_frac:.1f} \tau_x$')
                if n == 0:
                    if i == -1:
                        i = 'K' if n_frames else 'N'
                    ax.set_ylabel(fr'${symbol}_{i}$')


        ax = plt.Subplot(fig, inner_gs[2, 0])
        ax.set_xticks([])
        ax.set_yticks([])
        remove_border(ax)
        ax.set_ylabel(r'$\dots$')
        fig.add_subplot(ax)

    n_cols = t_mult * n_frames
    fig = plt.figure(figsize=(12, 8))
    main_gs = fig.add_gridspec(3, n_cols, hspace=.5)

    gs_X = fig.add_subplot(main_gs[0, :])
    gs_R = fig.add_subplot(main_gs[1, :])
    gs_A = fig.add_subplot(main_gs[2, :])

    populate_sgs(gs_X, X, 'x', 'Data')
    populate_sgs(gs_R, R, r'\hat x', 'Reconstruction')
    populate_sgs(gs_A, A, 'A', 'Dictionary', plot_A=True)
def plot_dict(A, im_shape, nrow=3, ncol=4, order=True):
    A_sub = A.T[:nrow*ncol]
    if order is True:
        order = (-np.linalg.norm(A_sub, axis=1)).argsort()
        A_sub = A_sub[(-np.linalg.norm(A_sub, axis=1)).argsort()]
    cmax = A_sub.max()
    cmin = A_sub.min()
    max_size = 8
    if ncol > nrow:
        W = 8
        H = 8 * nrow / ncol
    else:
        H = 8
        W = 8 * ncol / nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(W, H))
    axes = [ax for row in axes for ax in row]
    for n, a in enumerate(A_sub):
        axes[n].imshow(a.reshape(im_shape), clim=(cmin, cmax), cmap='Greys_r')
        axes[n].set_xticks([])
        axes[n].set_yticks([])
def animate_dict(A, n_frames, ratio=1.5, order='auto', out=None):
    n_time, n_dim, n_dict = A.shape
    skip = n_time // n_frames
    if order == 'auto':
        order = (-np.linalg.norm(A[-1], axis=0)).argsort()
    elif order is None:
        order = np.arange(n_dict)

    cmin = -min(-A.min(), A.max())
    cmax = -cmin

    nrows = ((n_dict/ratio)**0.5)
    ncols = int(np.ceil(n_dict / nrows))
    nrows = int(np.ceil(nrows))

    fig, axes = plt.subplots(nrows, ncols)
    axes = [ax for row in axes for ax in row]
    img_plots = []
    for ax in axes:
        img_plots.append(ax.imshow([[0]], clim=(cmin, cmax), cmap='gray'))
        ax.set_xticks([])
        ax.set_yticks([])
    
    def animate(n):
        print(n)
        for i, a in enumerate(A[n * skip].T[order]):
            a = a.reshape((8, 8))
            img_plots[i].set_data(a)
        fig.suptitle(n)

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=True)
    if out is None:
        plt.show()
    else:
        anim.save(out)

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
