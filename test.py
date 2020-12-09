import torch
import numpy

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