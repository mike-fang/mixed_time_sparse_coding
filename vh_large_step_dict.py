from vh_patches import *

# DSC params
dsc_params = dict(
    n_A = 5,
    n_s = 200,
    eta_A = 5,
    eta_s = .01,
)
DIM = 8
OC = 2

PI = 0.05

EXP = 'asynch'
assert EXP in ['dsc', 'ctsc', 'asynch', 'lsc']
base_dir = f'vh_large_{EXP}'

loader, model_params, solver_params = vh_loader_model_solver(dim=DIM, batch_frac=2, dict_oc=OC, dsc_params=dsc_params, pi=PI, exp=EXP)

# Define model, solver
model = CTSCModel(**model_params)
#solver = CTSCSolver(model, **solver_params)

solver = CTSCSolver(model, **solver_params)
dir_path = solver.get_dir_path(base_dir)
print(solver.t_max)
soln = solver.solve(loader, soln_T=1)
solver.save_soln(soln)

X = soln['x'][:]
R = soln['r'][:]
A = soln['A'][:]

print(A)
assert False

out_path = os.path.join(dir_path, 'evol.mp4')
out_path = None
show_img_XRA(X, R, A, img_shape=(DIM, DIM), out_file=out_path)
