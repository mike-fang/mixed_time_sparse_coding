from mt_sc_direct_implementation import MixT_SC, save_soln
from loaders import HVLinesLoader
import matplotlib.pylab as plt
import numpy as np
from matplotlib import animation

H = W = 5
p = .2
n_batch = 10

tau_s = 1e1
tau_x = 1e2
tau_A = 1e3
T_RANGE = 1e4
T_STEPS = int(T_RANGE)
tspan = np.linspace(0, T_RANGE, T_STEPS, endpoint=False)

n_dim = int(H * W)
n_sparse = n_dim * 2

l1 = .5
l0 = .5
sigma = 1

loader = HVLinesLoader(H, W, n_batch, p=p)

mtsc = MixT_SC(n_dim, n_sparse, tau_s, tau_x, tau_A, l0, l1, sigma, n_batch, positive=False)

print(loader.bases.shape)
def sA_HV_bases():
    s = np.zeros((n_batch, n_sparse))
    A = np.zeros((n_dim, n_sparse))
    A[:, :H + W] = loader.bases.T
    return s, A

s, A = sA_HV_bases()
solns = mtsc.solve(loader, tspan, init_sA=sA_HV_bases)
save_soln(solns, './results/hv_line_HV_init.h5py')


n_row = 5
n_col = 10
n_frames = 10
skip = int(T_RANGE//n_frames)
# Initialize all imshow plots
fig, axes = plt.subplots(5, 10)

img_plot = []
for ax_row in axes:
    for ax in ax_row:
        img_plot.append(
                ax.imshow([[0]])
                )
        ax.set_xticks([])
        ax.set_yticks([])


def animate(n):
    A = solns['A'][n*skip]
    A = A.reshape((W, H, -1))
    for i in range(len(A[0, 0, :])):
        a = A[:, :, i]
        img_plot[i].set_data(a)
        img_plot[i].autoscale()
    fig.suptitle(n)
anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=100, repeat=True)
plt.show()
