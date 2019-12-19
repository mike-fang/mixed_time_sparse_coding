from mtsc import *

def train_dsc():
    model_params = dict(
            n_dict = 3,
            n_dim = 2,
            n_batch = N_BATCH,
            positive = True
            )
    solver_params = [
            dict(params = ['s_data'], tau=LR_S**(-1), T=0),
            dict(params = ['A'], tau=LR_A**(-1)),
            ]
    init = dict(
            pi = PI,
            l1 = L1,
            sigma = 1,
            )
    dsc_solver = DSCSolver(model_params, solver_params)
    dsc_solver.model.reset_params(init=init)
    dsc_solver.set_loader(loader, tau_x)
    soln = dsc_solver.start_new_soln(tmax=tmax)
    show_2d_evo(soln)
def train_mtsc_alpha(alpha=None):
    model_params = dict(
            n_dict = 3,
            n_dim = 2,
            n_batch = N_BATCH,
            positive = True
            )
    solver_params = [
            dict(params = ['s_data'], tau=LR_S**(-1), T=0),
            dict(params = ['A'], tau=LR_A**(-1)),
            ]
    init = dict(
            pi = PI,
            l1 = L1,
            sigma = 1,
            )
    mtsc_solver = MTSCSolver(model_params, solver_params, alpha=alpha)
    mtsc_solver.model.reset_params(init=init)
    mtsc_solver.set_loader(loader, tau_x)
    soln = mtsc_solver.start_new_soln(tmax=tmax)
    show_2d_evo(soln)

if __name__ == '__main__':

    PI = 1
    L1 = 2
    N_BATCH = 3
    new = True
    tmax = int(5e2)
    tau_x = int(1e2)
    LR_S = .5
    LR_A = 0.005
    TAU_A = LR_A * tau_x
    loader = StarLoader(n_basis=3, n_batch=N_BATCH, sigma=0, pi=.2, l1=.5)
    loader = StarLoader_(n_basis=3, n_batch=3, sigma=.4, shuffle=False)
    train_mtsc_alpha(alpha=.1)
