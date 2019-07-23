import torch as th
from time import time
import random

N = int(1e4)

dW = th.zeros(N)
a = 0

t0 = time()
dW.normal_()
print(dW)

print(time() - t0)
for dw in dW:
    a += dw
print(a)

print(time() - t0)

print('=============')

t0 = time()


a = 0
for dw in dW:
    a += random.random()
print(a)

print(time() - t0)
