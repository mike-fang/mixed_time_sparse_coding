import torch as th
import numpy as np
from ctsc import *
from loaders import VanHaterenSampler
from visualization import show_img_XRA
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis

def vh_loader_model_solver(dim, batch_frac, dict_oc, pi, exp, dsc_params):
    H = W = dim
    N_DIM = H * W
    N_BATCH = int(N_DIM * batch_frac)
    N_DICT = int(dict_oc * N_DIM)
    loader = VanHaterenSampler(H, W, N_BATCH)
    if exp != 'lsc':
        pi = 1

    model_params = dict(
            n_dict=N_DICT,
            n_dim=N_DIM,
            n_batch=N_BATCH,
            positive=True,
            pi=pi,
            l1=1.0,
            sigma=1.0,
            )
    solver_params = dsc_solver_param(**dsc_params)
    if exp == 'dsc':
        pass
    elif exp == 'ctsc':
        solver_params['spike_coupling'] = False
    elif exp == 'asynch':
        solver_params['spike_coupling'] = False
        solver_params['asynch'] = True
    elif exp == 'lsc':
        solver_params['spike_coupling'] = False
        solver_params['asynch'] = True
        solver_params['T_u'] = 1
    return loader, model_params, solver_params

if __name__ == '__main__':
    N_S = 50
    # DSC params
    dsc_params = dict(
        n_A = 4000,
        n_s = N_S,
        eta_A = 0.02,
        eta_s = 0.1,
    )
    DIM = 8
    OC = 2

    PI = 0.05

    EXP = 'dsc'
    LOAD = False
    assert EXP in ['dsc', 'ctsc', 'asynch', 'lsc']
    base_dir = f'vh_dim_{DIM}_{EXP}'

    loader, model_params, solver_params = vh_loader_model_solver(dim=DIM, batch_frac=0.5, dict_oc=OC, dsc_params=dsc_params, pi=PI, exp=EXP)

    # Define model, solver
    model = CTSCModel(**model_params)
    #solver = CTSCSolver(model, **solver_params)

    # Load or make soln
    if LOAD:
        dir_path = get_timestamped_dir(load=True, base_dir=base_dir)
        soln = h5py.File(os.path.join(dir_path, 'soln.h5'))
    else:
        solver = CTSCSolver(model, **solver_params)
        dir_path = solver.get_dir_path(base_dir)
        soln = solver.solve(loader, soln_T=N_S, soln_offset=-1)
        solver.save_soln(soln)

    X = soln['x'][:]
    R = soln['r'][:]
    A = soln['A'][:]

    out_path = os.path.join(dir_path, 'evol.mp4')
    show_img_XRA(X, R, A, n_frames=1e2, img_shape=(DIM, DIM), out_file=out_path)
