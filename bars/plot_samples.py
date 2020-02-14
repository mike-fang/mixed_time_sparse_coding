from context import *

def plot_samples(pi=.3, l1=1, sigma=0):
    H = W = 8
    loader = BarsLoader(H, W, 16, l1=l1, p=pi, sigma=sigma, numpy=True)
    x = loader().reshape((-1, 8, 8))
    fig, axes = plt.subplots(2, 8, figsize=(8, 2))
    axes = [ax for row in axes for ax in row]
    for n, ax in enumerate(axes):
        ax.imshow(x[n], cmap='Greys_r')
        #ax.set_xlabel(rf'$A_{{{n}}}$')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(fr'Bars Samples: $\lambda = {l1}, \pi = {pi}, \sigma = {sigma}$')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_dict():
    H = W = 8
    loader = BarsLoader(H, W, 16,  numpy=True)
    x = loader.bases.reshape((-1, 8, 8))
    fig, axes = plt.subplots(2, 8, figsize=(8, 2))
    axes = [ax for row in axes for ax in row]
    for n, ax in enumerate(axes):
        ax.imshow(x[n], cmap='Greys_r')
        #ax.set_xlabel(rf'$A_{{{n}}}$')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('Bars Dictionary')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

plot_dict()
plt.show()
