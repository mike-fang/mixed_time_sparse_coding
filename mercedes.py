from ctsc import *
from loaders import StarLoader
from visualization import *

SIGMA = 0
N_BATCH = 3
loader = StarLoader(n_basis=3, n_batch=N_BATCH, sigma=0, pi=.5, l1=.5)
model_params = dict(
        n_dict = 3,
        n_dim = 2,
        n_batch = N_BATCH,
        positive = True,
        l1 = .5,
        pi = .9,
        sigma = 1,
        )
solver_params = dict(
        tau_A = 5e3,
        tau_u = 1e2,
        tau_x = 1e9,
        )

model = CTSCModel(**model_params)
solver = CTSCSolver(model, loader, **solver_params)
tspan = np.arange(1e4)
soln = solver.solve(tspan, soln_N=1e2)

show_2d_evo(soln)
