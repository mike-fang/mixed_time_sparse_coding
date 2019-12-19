from mtsc import *

PI = .5
L1 = .05
SIGMA = 0
N_BATCH = 3
new = True
tmax = int(1e4)
tau_x = int(1e3)
tau_s = int(1e3)
tau_A = 5e5
loader = StarLoader(n_basis=3, n_batch=N_BATCH, sigma=SIGMA, pi=PI, l1=L1)
model_params = dict(
        n_dict = 3,
        n_dim = 2,
        n_batch = N_BATCH,
        positive = True
        )
solver_params = [
        dict(params = ['s_data'], tau=tau_s, T=1),
        #dict(params = ['s_model'], tau=-tau_s/5, T=0),
        #dict(params = ['x_model'], tau=-tau_x/5, T=1),
        dict(params = ['A'], tau=tau_A),
        ]
init = dict(
        pi = PI,
        l1 = L1,
        sigma = 1,
        )
try:
    if new:
        assert False
    soln =Solutions.load()
except:
    mtsc_solver = MTSCSolver(model_params, solver_params)
    mtsc_solver.model.reset_params(init=init)
    mtsc_solver.set_loader(loader, tau_x)
    soln = mtsc_solver.start_new_soln(tmax=tmax, n_soln=1000)

show_2d_evo(soln)
