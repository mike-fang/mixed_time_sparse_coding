import numpy as np
import torch as th

p = th.arange(10.)/10
p = p[:, None]

where_load = th.zeros(10, 3).bernoulli_(p)
print(p)
print(where_load)
