import numpy as np
from loaders import BarsLoader
from tqdm import tqdm
import matplotlib.pylab as plt

H = W = 4
N_DIM = H * W
N_BATCH = H * W
N_DICT = H + W
PI = 0.3
loader = BarsLoader(H, W, N_BATCH, p=PI, test=False)

N_A = 2500
N_S = 10
eta_A = 0.1
eta_S = 0.1

U0 = 0.20

class LCA:
    def __init__(self, n_dim, n_dict, n_batch, u0=1, positive=True):
        self.n_dim = n_dim
        self.n_dict = n_dict
        self.n_batch = n_batch
        self.u0 = u0
        self.positive = positive
        self.reset_params()
    def reset_params(self):
        self.A = np.random.normal(0, 1, size=(self.n_dim, self.n_dict))
        self.u = np.random.normal(0, 1, size=(self.n_dict, self.n_batch))
    @property
    def s(self):
        where_thresh = np.abs(self.u) <= self.u0
        s = np.zeros_like(self.u)
        s[~where_thresh] = self.u[~where_thresh] - self.u0 * np.sign(self.u[~where_thresh])
        if self.positive:
            s = np.abs(s)
        return s
    def step_u(self, x, eta):
        sign = np.sign(self.u) if self.positive else 1
        grad = self.A.T @ (self.A @ self.s - x.T) * sign + (self.u - self.s)
        self.u -= eta * grad
    def step_A(self, x, eta, normalize=True):
        grad = (self.A @ self.s - x.T) @ self.s.T
        self.A -= eta * grad / self.n_batch
        if normalize:
            self.A /= np.linalg.norm(self.A, axis=0)
    def rmse(self, x):
        err = ((self.A @ self.s - x)**2).mean()**0.5
        return err

lca = LCA(n_dim=N_DIM, n_dict=N_DICT, n_batch=N_BATCH, u0=U0, positive=True)
lca.A *= 0
lca.A[:N_DICT, :N_DICT] = np.eye(N_DICT)
lca.u *= 0
lca.u += 1

for n in tqdm(range(N_A)):
    x = np.array(loader())
    for k in range(N_S):
        lca.step_u(x, eta_S)
    lca.step_A(x, eta_A)

A = lca.A
fig, axes = plt.subplots(nrows=2, ncols=4)
axes = [a for row in axes for a in row]
for n, ax in enumerate(axes):
    ax.imshow(A[:, n].reshape(H, W))
plt.show()

