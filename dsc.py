import torch as th
import numpy as np
from loaders import BarsLoader
from torch.nn import Parameter
from tqdm import tqdm
import matplotlib.pylab as plt

H = W = 4
N_DIM = H * W
N_BATCH = H * W
N_DICT = H + W
PI = 0.3
loader = BarsLoader(H, W, N_BATCH, p=PI, test=True)

N_A = 4
N_S = 10
eta_A = 0.1
eta_S = 0.1

sigma = 1.0
l1 = 0.2
l1 = 0


A = Parameter(th.Tensor(N_DIM, N_DICT))
s = Parameter(th.Tensor(N_DICT, N_BATCH))
A.data.normal_()
s.data.normal_()

s.data *= 0
s.data += 1
A.data *= 0
A.data[:N_DICT, :N_DICT] = th.eye(N_DICT)

plt.imshow(A.data.numpy())
plt.show()
def energy(A, s, x):
    s = th.abs(s)
    recon = 0.5 * ((A@s - x.t())**2).sum()
    sparse = th.abs(s).sum()
    return recon/sigma**2 + l1 * sparse
def dEds(A, s, x):
    return A.t() @ (A@th.abs(s) - x.t()) * th.sign(s) + l1 * th.sign(s).sum()


def zero_grad(p):
    if p.grad is not None:
        p.grad.detach_()
        p.grad.zero_()

for n in tqdm(range(N_A)):
    x = loader()
    for k in range(N_S):
        zero_grad(s)
        E = energy(A, s, x)
        E.backward()
        s.data.add_(-eta_S, s.grad)
    zero_grad(A)
    E = energy(A, s, x)
    E.backward()
    A.data.add_(-eta_A/N_BATCH, A.grad)
    A.data /= A.norm(dim=0)
    plt.imshow(A.data.numpy())
    plt.show()
assert False
A = A.data.numpy()
fig, axes = plt.subplots(nrows=2, ncols=4)
axes = [a for row in axes for a in row]
for n, ax in enumerate(axes):
    ax.imshow(A[:, n].reshape(H, W))
plt.show()
