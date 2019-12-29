import torch as th
from torch.nn import Parameter


U = Parameter(th.tensor(5.))
X = (U.clone().data.numpy())
print(X)
U.data += 1
print(X)
