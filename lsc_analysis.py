from soln_analysis import SolnAnalysis
import numpy as np
from ctsc import get_timestamped_dir
import matplotlib.pylab as plt
import seaborn as sns

exp = 'dsc'

dir_path = get_timestamped_dir(load=True, base_dir=f'vh_dim_8_{exp}')
analysis = SolnAnalysis(dir_path, lca=False)

time = analysis.time
mse = np.mean(analysis.mse(), axis=1)
energy = analysis.energy()
mean_nz = analysis.mean_nz()

if False:
    diff = analysis.diff()
    D0 = (diff[::10, 0, :])
    D1 = (diff[::10, 2, :])

    g = sns.jointplot(D0, D1, kind='kde')
    g.ax_marg_x.set_xlim(-2, 2)
    g.ax_marg_y.set_ylim(-2, 2)
    plt.savefig(f'./figures/{exp}_joint_distr.pdf')
    plt.show()
else:
    plt.subplot(211)
    plt.plot(time[10:], mse[10:])
    plt.subplot(212)
    plt.plot(time[10:], energy[10:])
    plt.savefig(f'./figures/{exp}_err_distr.pdf')
    plt.show()
