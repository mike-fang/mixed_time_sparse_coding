import torch as th
import numpy as np
from euler_maruyama import EulerMaruyama
from torch.nn import Parameter
from tqdm import tqdm
import matplotlib.pylab as plt

ratio = 20
tau_s = ratio
tau_x = int(ratio**0.5)
#tau_x = ratio**3
steps = ratio**3 * 4

x = Parameter(th.tensor(5.))
s = Parameter(th.tensor(0.))

param_groups = [
        {'params': [s], 'tau': tau_s, 'mu': 0, 'T': 1},
        ]
solver = EulerMaruyama(param_groups, dt=1)

def closure():
    energy = (s - x)**2/2
    energy.backward()
    return energy

X = []
S = []
for n in tqdm(range(steps)):
    solver.zero_grad()
    if n % tau_x == 0:
        x.data *= -1
    solver.step(closure)
    X.append(float(x))
    S.append(float(s))

plt.subplot(211)
plt.plot(S)
plt.plot(X)
plt.subplot(212)
S_range = np.linspace(-10, 10, 100)
p = np.exp(-(S_range-5)**2/2)
p += np.exp(-(S_range+5)**2/2)
p /= p.sum() * (S_range[1] - S_range[0])

plt.hist(S, bins=50, density=True)
plt.plot(S_range, p)
plt.show()



