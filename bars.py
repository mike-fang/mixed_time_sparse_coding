from loaders import BarsLoader
import matplotlib.pylab as plt
import numpy as np
import torch as th
from visualization import show_img_evo, show_img_XRA
from mtsc import *
from L0_sparsity import *

if __name__ == '__main__':
    #Define loader
    H = W = 4
    n_batch = H * W
    p = .5
    p_loader = p
    TAU_S = int(1e2)
    TAU_X = int(1e3)
    TAU_A = int(5e3)
    TAU_RATIO = 10
    ALPHA = 0.2
    OC = 1
    n_dict = int(H * W * OC)
    n_dict = H + W
    tmax = 1e4
    ASYNCH = True

    loader = BarsLoader(H, W, n_batch, p=p_loader)
    model_params = dict(
            n_dict = n_dict,
            n_dim = H * W,
            n_batch = n_batch,
            positive = True
            )
    solver_params = [
            dict(params = ['s_data'], tau=TAU_S, T=1),
            dict(params = ['A'], tau=TAU_A*n_batch),
            ]
    init = dict(
            l1 = 1,
            pi = p,
            sigma = 1,
            A = loader.bases.T,
            )


    mtsc_solver = MTSCSolver(model_params, solver_params, base_dir='hv_lines', im_shape=(H, W))
    mtsc_solver.model.reset_params(init=init)
    mtsc_solver.set_loader(loader, tau_x=TAU_X)
    soln = mtsc_solver.start_new_soln(tmax=tmax, n_soln=None, t_save=None, rand_tau_x=ASYNCH)

    dir_path = mtsc_solver.dir_path
    #plt.plot(soln['energy'])
    #plt.savefig(os.path.join(dir_path, 'energy.pdf'))

    reshaped_params = soln.get_reshaped_params()
    X = reshaped_params['x_data']
    R = reshaped_params['r_data']
    A = reshaped_params['A']
    XRA = np.concatenate((X, R, A), axis=1)

    out_path = os.path.join(dir_path, 'evol.mp4')
    #out_path = None
    show_img_XRA(X, R, A, n_frames=100, out_file=out_path)
    #plot_sparsity(dir_path=mtsc_solver.dir_path, pi=p, out=False)
    plot_sparsity(dir_path=None, pi=p, out=False)
