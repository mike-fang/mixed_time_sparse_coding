from soln_analysis import SolnAnalysis
import numpy as np
from ctsc import get_timestamped_dir
import matplotlib.pylab as plt
import seaborn as sns
import matplotlib

font = {'family' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

def plot_density(analysis, name, color, thresh=2e-2):
    analysis.zero_coeffs(thresh)
    mean_nz = analysis.mean_nz()
    time = analysis.time
    plt.plot(time, mean_nz, 'k', color=color, label=name)
def plot_nz_distr(out=False):
    colors = ['k', 'b', 'g']
    exps = ['dsc', 'lsc', 'lca']
    #fig = plt.figure(figsize=(8, 3))
    for c, exp in zip(colors, exps):
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        base_dir = f'bars_{exp}'
        dir_path = get_timestamped_dir(load=True, base_dir=base_dir, dir_name='random_dict')
        analysis = SolnAnalysis(dir_path)

        if exp != 'lsc':
            analysis.zero_coeffs()
        analysis.plot_nz_hist(s_max=6, title=None, n_bins=30, log=False, eps_s=5e-2, ylim=(0, 1.5))

        plt.title('Random Dictionary')
        plt.subplot(122)
        dir_path = get_timestamped_dir(load=True, base_dir=base_dir, dir_name='learned_dict')
        analysis = SolnAnalysis(dir_path)

        if exp != 'lsc':
            analysis.zero_coeffs()
        analysis.plot_nz_hist(s_max=6, title=None, n_bins=30, log=False, eps_s=5e-2, ylim=(0, 1.5))

        plt.subplots_adjust(hspace=.5)
        plt.title('Trained Dictionary')
        plt.tight_layout(pad=0)
        if out:
            plt.savefig(f'./figures/bars_distr_{exp}.pdf', bb_inches='tight')
        plt.show()
def plot_density_evo(exp_colors, save=True):
    fig = plt.figure(figsize=(8, 3))
    for exp in exp_colors:
        c = exp_colors[exp]
        base_dir = f'bars_{exp}'
        dir_path = get_timestamped_dir(load=True, base_dir=base_dir)
        analysis = SolnAnalysis(dir_path)
        plot_density(analysis, name=exp, color=c, thresh=None)
    time = analysis.time
    plt.plot(time, np.ones_like(time) * 0.3, 'm--', label=r'$\pi = 0.3$')
    plt.xlim(0, 1e4)
    plt.legend()
    plt.ylabel('Density')
    plt.xlabel('Time')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig(f'./figures/bars_density.pdf')
    plt.show()
if __name__ == '__main__':
    exp_colors = {
            'dsc' : 'k',
            'lca' : 'r',
            'lsc' : 'b'
            }
    plot_nz_distr(out=True)
    #plot_nz_distr('bars_untrained')
    #plot_density_evo(exp_colors, save=False)
