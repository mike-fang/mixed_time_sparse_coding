import torch as th


X = th.tensor(-10.)
def softplus(x):
    return th.nn.functional.softplus(x)
def inv_sp(x):
    return th.log(th.exp(x) - 1)

u = th.nn.functional.softplus(X)
print(u)
print(inv_sp(u))
