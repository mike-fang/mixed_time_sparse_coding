from mtsc import *

def train_dsc(tmax, tau_x, model, solver, loader, t_start=0, n_soln=None, out_dir=None, t_save=None, normalize_A=True):
    tmax = int(tmax)
    tau_x = int(tau_x)
    t_start = int(t_start)
    if (n_soln is None) or (n_soln > tmax):
        n_soln = tmax
    when_out = np.linspace(t_start, tmax+1, num=n_soln+1, dtype=int)[:-1]
    
    # If t_save not specified, save only begining and end
    if (out_dir is not None) and (t_save is None):
        t_save = tmax

    # Define soln
    soln = defaultdict(list)

    # train model
    #x = loader()
    def closure():
        energy = model(x)
        energy.backward()
        return energy

    for p in solver.param_groups:
        try:
            if model.A in p['params']:
                A_group = p
        except:
            pass
        try:
            if model.s_data in p['params']:
                s_group = p
        except:
            pass
    for t in tqdm(range(t_start, tmax + 1)):
        if t % int(tau_x) == 0:
            x = loader().T
            A_group['coupling'] = 1
            s_group['coupling'] = 0
        else:
            A_group['coupling'] = 0
            s_group['coupling'] = 1
        solver.zero_grad()
        energy = float(solver.step(closure))
        if normalize_A:
            model.A.data = model.A / model.A.norm(dim=0)
            #A = model.A.data.numpy()
            #_, S, _ = np.linalg.svd(A)
        if t in when_out:
            for n, p in model.named_parameters():
                soln[n].append(p.clone().data.cpu().numpy())
            soln['r_model'].append(model.get_recon(model.s_model).data.numpy())
            soln['r_data'].append(model.get_recon(model.s_data).data.numpy())
            soln['x_data'].append(x.data.numpy())
            soln['t'].append(t)
            soln['energy'].append(energy)
        if (t_save is not None) and (t % t_save == 0):
            save_checkpoint(model, solver, t, out_dir)

    for k, v in soln.items():
        soln[k] = np.array(v)
    return soln

if __name__ == '__main__':
    PI = 1
    L1 = 2
    N_BATCH = 3
    new = True
    tmax = int(5e3)
    tau_x = int(1e2)
    LR_S = 1
    LR_A = 0.02
    #loader = StarLoader(n_basis=3, n_batch=N_BATCH, sigma=0, pi=.8, l1=.5)
    loader = StarLoader_(n_basis=3, n_batch=3, sigma=.4, shuffle=False)
    model_params = dict(
            n_dict = 3,
            n_dim = 2,
            n_batch = N_BATCH,
            positive = True
            )
    solver_params = [
            dict(params = ['s_data'], tau=LR_S**(-1)),
            dict(params = ['A'], tau=LR_A**(-1)),
            ]
    init = dict(
            pi = PI,
            l1 = L1,
            sigma = 1,
            )
    model, solver = get_model_solver(model_params=model_params, solver_params=solver_params)
    model.reset_params(init=init)
    soln = train_dsc(tmax, tau_x, model, solver, loader)
    show_2d_evo(soln)

    assert False
    try:
        if new:
            assert False
        soln = Solutions.load()
    except:
        mtsc_solver = MTSCSolver(model_params, solver_params)
        mtsc_solver.model.reset_params(init=init)
        mtsc_solver.set_loader(loader, tau_x)
        soln = mtsc_solver.start_new_soln(tmax=tmax, n_soln=1000)

    show_2d_evo(soln)

