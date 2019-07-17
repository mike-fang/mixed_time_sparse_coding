from mt_sc import MixT_SC
import torch as th
import numpy as np
import matplotlib.pyplot as plt
tau_s = 1e-2
tau_x = 1
tau_A = 4e10
sigma = 1e20
l1 = 2
l0 = .3
#s0 = 3.5
s0 = -np.log(1 - l0) / l1
print(s0)

T_RANGE = 2e1
T_STEPS = int(1e4)

tspan = th.linspace(0, T_RANGE, T_STEPS)

X = th.tensor([[0., 0]])

mtsc = MixT_SC(tau_s, tau_x, tau_A, sigma, l1, s0)

s_soln, A_soln = mtsc.train(X, tspan, n_sparse=2)

s_soln = s_soln.data.numpy()

print((np.abs(s_soln) < s0).mean(0))
u_soln = (s_soln - s0* np.sign(s_soln) ) * (np.abs(s_soln) > s0)
print((u_soln == 0).mean())


plt.subplot(121)
plt.scatter(*u_soln.T, s=1)
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.subplot(122)
x_ = 10
u = np.linspace(-x_, x_, 100)
plt.hist(u_soln[:, 0], bins=40, density=True)
p = l1 / 2 * np.exp(-l1 * (np.abs(u) + s0))
plt.plot(u, p)
plt.ylim([0, 1])
plt.xlim([-x_, x_])
plt.show()
