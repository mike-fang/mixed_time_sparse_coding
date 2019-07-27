import matplotlib.pylab as plt

n_row = 5
n_col = 10
fig, axes = plt.subplots(n_row, n_col, figsize=(10, 5))

idx = 0
img_plot = []
for ax_row in axes:
    for ax in ax_row:
        img_plot.append(
                ax.imshow([[0]])
                )
        ax.set_xticks([])
        ax.set_yticks([])

img_plot[2].set_data([[1, 0], [0, 1]])
img_plot[2].autoscale()
plt.show()

