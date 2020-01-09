from soln_analysis import SolnAnalysis
import numpy as np
from ctsc import get_timestamped_dir
import matplotlib.pylab as plt

dir_path = get_timestamped_dir(load=True, base_dir='vh_dim_8_lca')
analysis = SolnAnalysis(dir_path, lca=True)

time = analysis.time
psnr = np.mean(analysis.mse(), axis=1)
energy = analysis.energy()
mean_nz = analysis.mean_nz()

plt.subplot(211)
plt.plot(time, energy)
plt.title('Energy')
plt.plot(time, energy - psnr)
plt.xticks([])
plt.subplot(212)
plt.plot(time, 1 - mean_nz)
plt.title('L0 Sparsity')
plt.show()

