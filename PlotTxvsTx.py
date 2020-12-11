"""To better understand the piecewise function, plot T(x) vs T(x) wrt to two variations"""


import matplotlib.pyplot as plt
import json
from f_gan import Divergence
import torch

xvar='hellinger'
yvar='total_variation'


divx=Divergence(xvar)
divy=Divergence(yvar)

fig = plt.figure()
plt.ioff()
#If there is no file with a specific loss, just skip it

step=[i/10 for i in range(-50, 50)]
x=[]
y=[]
for s in step:
    s=torch.Tensor([s])
    x.append(divx.Tx(s).item())
    y.append(divy.Tx(s).item())


plt.plot(x, y)
plt.xlabel('Sampling Size')
plt.ylabel('KL divergence')
plt.legend()
plt.title('Error induced by the MC sampling algorithm in calculating the f-divergence')
plt.savefig(f"Gaussian_IMGS/HellingerVsTotalVar.png")
plt.close(fig)