from soln_analysis import SolnAnalysis
import matplotlib.pylab as plt

d = './results/bars_lsc/PI_0p10_pi_0p10/'
analysis = SolnAnalysis(d)
analysis.plot_nz_hist(last_frac=1, log=False)
plt.show()
