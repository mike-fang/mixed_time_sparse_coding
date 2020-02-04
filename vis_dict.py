import numpy as np
from plt_env import *

im_shape = (8, 8)
A = np.load('./oc2A.npy').T
A = A[np.linalg.norm(A, axis=0).argsort()]
n_dict, _ = A.shape

n_col = 8
n_row = int(n_dict // 8)

#fig, axes = plt.subplots(nrows=n_row, ncols=n_col)
#axes = [a for row in axes for a in row]

A_fft = 0
for n, a in enumerate(A):
    a_ = np.fft.fft2(a.reshape(im_shape))
    a_ = np.fft.fftshift(a_)
    a_mod = np.abs(a_)
    A_fft += a_mod
    if False:
        axes[n].imshow(a_mod.reshape((8, 8)), cmap='Greys_r')
        axes[n].set_xticks([])
        axes[n].set_yticks([])
plt.show()
plt.imshow(A_fft, cmap='Greys_r')
plt.show()
