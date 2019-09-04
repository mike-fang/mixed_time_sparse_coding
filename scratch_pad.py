import torch as th
from torch.nn import Parameter

p = Parameter(th.zeros(5))
data = th.zeros(3, 5)
data[0] = p
p.data += 1
data[1] = p

print(data)
