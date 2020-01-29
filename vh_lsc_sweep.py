import torch as th
import numpy as np
from ctsc import *
from loaders import VanHaterenSampler
from visualization import show_img_XRA
import matplotlib.pylab as plt
from soln_analysis import SolnAnalysis

if __name__ == '__main__':
    DIM = 8
    OC = 2
    BATCH_FRAC = 2
    H = W = DIM
    N_DIM = H * W
    N_BATCH = int(N_DIM * BATCH_FRAC)
    N_DICT = int(OC * N_DIM)
    PI = 0.05
    EXP = 'lsc'
    LOAD = False

    N_S = 400
    # DSC params
    dsc_params = dict(
        n_A = 400,
        n_s = N_S,
        eta_A = 0.2,
        eta_s = 0.05,
    )
    if EXP != 'lsc':
        PI = 1

    for PI in np.arange(0.05, 0.5, 0.05):
        for L1 in np.arange(2, 3, 0.5):
            PI = round(float(PI), 2)
            L1 = round(float(L1), 2)
            NAME = f'l1_{L1}_p1_{PI}'.replace('.', 'p')
            model_params = dict(
                    n_dict=N_DICT,
                    n_dim=N_DIM,
                    n_batch=N_BATCH,
                    positive=True,
                    pi=PI,
                    l1=L1,
                    sigma=1.0,
                    )

            assert EXP in ['dsc', 'ctsc', 'asynch', 'lsc']
            base_dir = f'vh_dim_{DIM}_{EXP}'
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
            solver = CTSCSolver(model, **solver_params)
            try:
                dir_path = solver.get_dir_path(base_dir, name=NAME, overwrite=False)
            except:
                continue
            soln = solver.solve(loader, soln_T=N_S, soln_offset=-1)
            solver.save_soln(soln)

            continue
            X = soln['x'][:]
            R = soln['r'][:]
            A = soln['A'][:]

            out_path = os.path.join(dir_path, 'evol.mp4')
            show_img_XRA(X, R, A, n_frames=1e2, img_shape=(DIM, DIM), out_file=out_path)
