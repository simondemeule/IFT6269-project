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

plot_mural("MNIST", ["jensen_shannon", "total_variation"], [0, 0], 50, 5)