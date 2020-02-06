import numpy as np
import torch as th

s = th.FloatTensor(100000)
s.exponential_()
print(s.mean())
