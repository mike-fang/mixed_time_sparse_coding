import torch as th
from torch.nn import Parameter

x = Parameter(th.zeros(3, 3))
y = th.zeros(3, 3)
y.add_(3)
print(type(x))
print(type(x.data))
x.data.add_(5, 3)

print(x)
eta = th.FloatTensor(*x.shape).normal_()
print(eta)
