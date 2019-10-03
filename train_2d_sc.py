from mtsc import *

PI = .5
L1 = 1
SIGMA = 2
model_params = dict(
        n_dict = 3,
        n_dim = 2,
        n_batch = 3,
        positive = True
        )
solver_params = [
        dict(params = ['s'], tau=1e2, T=1),
        dict(params = ['A'], tau=5e4),
        ]
t_max = int(1e3)
tau_x = int(1e3)
#loader = StarLoader(n_basis=3, n_batch=model_params['n_batch'], pi=PI, l1=L1, sigma=SIGMA/5, positive=True)
loader = StarLoader_(n_basis=3, n_batch=model_params['n_batch'], sigma=2)
print(loader())
init = dict(
        pi = PI,
        l1 = L1,
        sigma = SIGMA,
        #A = loader.A
        )


if True:
    mtsc_solver = MTSCSolver(model_params, solver_params, noise_cache=0)
    mtsc_solver.model.reset_params(init=init)
    mtsc_solver.set_loader(loader, tau_x)
    soln = mtsc_solver.start_new_soln(tmax=t_max, n_soln=10000)
else:
    mtsc_solver = MTSCSolver.load()
    mtsc_solver.load_checkpoint(chp_name='start')
    mtsc_solver.set_loader(loader, tau_x)
    soln = mtsc_solver.start_new_soln(tmax=t_max, n_soln=10000)
f_name = os.path.join(mtsc_solver.dir_path, 'soln.mp4')
#f_name = None
show_2d_evo(soln, n_frames=100, f_out = f_name)
