from soln_analysis import *

def plot_energy(ds_name, exp_names, save_path=None, skip=1):
    if save_path == 'auto':
        save_path = f'./figures/energy_{ds_name}'

    colors = ['b', 'g', 'orange', 'r', 'violet']
    for n, exp in enumerate(exp_names):
        try:
            print(exp)
            dir_path = get_timestamped_dir(load=True, base_dir=f'{ds_name}_{exp}')
            analysis = SolnAnalysis(dir_path)
            time, energy, recon = analysis.energy(skip=skip)
            plt.plot(time, energy, c=colors[n], ls='-', label=f'{exp}: energy')
            plt.plot(time, recon, c=colors[n], ls=':', label=f'{exp}: recon')
        except:
            raise

    plt.yscale('log')
    plt.legend()
    plt.ylabel('Energy')
    plt.xlabel('Time')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    #plot_energy('vh', ['asynch', '1T'], save_path='auto', skip=1)
    ds_name = 'vh'
    exp = 'infer'
    dir_path = get_timestamped_dir(load=True, base_dir=f'{ds_name}_{exp}')
    analysis = SolnAnalysis(dir_path)
    tau_x = analysis.solver.tau_x
    analysis.set_soln(tau_x, -1)

    analysis.plot_nz_hist(log=False, n_bins=35, eps_p=None, ylim='auto')
    plt.savefig(f'./figures/{ds_name}_{exp}_nz_hist.pdf')
