import torch as th
import numpy as np
from ctsc import *
from loaders import VanHaterenSampler
from visualization import show_img_XRA
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis

if __name__ == '__main__':
    DIM = 8
    OC = 4
    BATCH_FRAC = 2
    H = W = DIM
    N_DIM = H * W
    N_BATCH = int(N_DIM * BATCH_FRAC)
    N_DICT = int(OC * N_DIM)
    PI = 0.30
    EXP = 'lsc'
    LOAD = False

    N_S = 400
    # DSC params
    dsc_params = dict(
        n_A = 1500,
        n_s = N_S,
        eta_A = 0.2,
        eta_s = 0.05,
    )
    if EXP != 'lsc':
        PI = 1

    PI = round(float(PI), 2)
    print(PI)
    model_params = dict(
            n_dict=N_DICT,
            n_dim=N_DIM,
            n_batch=N_BATCH,
            positive=True,
            pi=PI,
            l1=1.0,
            sigma=1.0,
            )

    assert EXP in ['dsc', 'ctsc', 'asynch', 'lsc']
    base_dir = f'vh_oc_{OC}_dim_{DIM}_{EXP}'
    loader = VanHaterenSampler(H, W, N_BATCH)

    solver_params = dsc_solver_param(**dsc_params)
    if EXP == 'dsc':
        pass
    elif EXP == 'ctsc':
        solver_params['spike_coupling'] = False
    elif EXP == 'asynch':
        solver_params['spike_coupling'] = False
        solver_params['asynch'] = True
    elif EXP == 'lsc':
        solver_params['spike_coupling'] = False
        solver_params['asynch'] = False
        solver_params['T_u'] = 1

    # Define model, solver
    model = CTSCModel(**model_params)

    # Load or make soln
    if LOAD:
        dir_path = get_timestamped_dir(load=True, base_dir=base_dir)
        soln = h5py.File(os.path.join(dir_path, 'soln.h5'))
    else:
        solver = CTSCSolver(model, **solver_params)
        dir_path = solver.get_dir_path(base_dir)
        soln = solver.solve(loader, soln_T=N_S, soln_offset=-1, normalize_A=False)
        solver.save_soln(soln)

    X = soln['x'][:]
    R = soln['r'][:]
    A = soln['A'][:]

    out_path = os.path.join(dir_path, 'evol.mp4')
    out_path = None
    show_img_XRA(X, R, A, n_frames=1e2, img_shape=(DIM, DIM), out_file=out_path)
