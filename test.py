import torch
import numpy
from plotting import *

"""
s = 5
m = []
n = []
for i in range(s):
    m.append(torch.rand(1, 2 * i, 2 * (i + s)))
    n.append(torch.rand(1, 2 * i, 2 * (i + s)))

p = torch.cat([torch.flatten(x) for x in m]).clone()
q = torch.cat([torch.flatten(x) for x in n])
d = numpy.linalg.norm(p - q)

print(d)
"""

#plot_mural("MNIST", ["jensen_shannon", "total_variation"], [0, 0], 50, 5)

#plot_walk_training('MNIST', 'total_variation', 15, show_plot=True)

#plot_mural("MNIST", ["hellinger", "hellinger"], [20, 19], 50, 10, epoch_shape_out=(5, 5))

plot_divergence_hyper("MNIST", "hellinger", [13, 14, 15, 16], ["128", "64", "32", "16"], "and Varying Batch Size", colors=["orangered", "crimson", "darkblue", "dodgerblue"])