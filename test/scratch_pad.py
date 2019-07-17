import numpy as np
import torch as th

n_data = 3
tau_x = 1
tspan = th.linspace(0, 10, int(1e2))

t_chunks = (tspan//tau_x).long()
uniq_idx = th.unique(t_chunks)
n_batches = np.ceil(len(th.unique(t_chunks))/n_data).astype(int)
x_idx = th.zeros_like(t_chunks)
for i in range(n_batches):
    batches = uniq_idx[i*n_data:(i+1)*n_data]
    rand_idx = th.randperm(n_data)
    for n, j in enumerate(batches):
        x_idx[t_chunks == j] = rand_idx[n]

return x_idx
