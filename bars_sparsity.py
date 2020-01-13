from soln_analysis import SolnAnalysis
import numpy as np
from ctsc import get_timestamped_dir
import matplotlib.pylab as plt
import seaborn as sns

def plot_density_mse(analysis, name, color, thresh=2e-2):
    analysis.zero_coeffs(thresh)
    mse = analysis.mse(mean=True)[20:]
    mean_nz = analysis.mean_nz()[20:]
    time = analysis.time[20:]
    plt.subplot(211)
    plt.plot(time, mean_nz, 'k', color=color, label=name)
    plt.plot(time, np.ones_like(time) * 0.3, 'r--')
    plt.legend()
    plt.xticks([])
    plt.title('Density')
    plt.subplot(212)
    plt.plot(time, -np.log(mse), color=color, label=name)
    plt.legend()
    plt.title('Negative Log MSE')

if __name__ == '__main__':
    colors = ['g', 'b', 'r','m']
    exps = ['dsc', 'ctsc', 'asynch', 'lsc']
    for c, exp in zip(colors, exps):
        base_dir = f'bars_{exp}'
        dir_path = get_timestamped_dir(load=True, base_dir=base_dir)
        analysis = SolnAnalysis(dir_path)
        plot_density_mse(analysis, name=exp, color=c)
    plt.savefig(f'./figures/ctsc_dsc_compare.pdf')
    plt.show()
