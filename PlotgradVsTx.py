"""Plot the value of the gradient of the discriminator and generator vs the value of T(x)"""
import torch
from f_gan import Divergence
import matplotlib.pyplot as plt
import numpy as np

fprim1=[0,1,-1,0,0,0]

variable2=torch.zeros(1, requires_grad=True)

x=[i/10 for i in range (-50, 50, 1)]
y=[]

fig = plt.figure()
plt.ioff()
# zip(['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon'],['blue','orange','purple','green','pink','gray'])
for loss,color,fprim in zip(['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon'],['blue','orange','purple','green','pink','gray'], fprim1):
    y = []
    div = Divergence(loss)
    variable = torch.zeros(1, requires_grad=True)
    for i in x:
        variable = torch.zeros(1, requires_grad=True) + i
        # variable2 = torch.zeros(1, requires_grad=True)+j
        lg = div.G_loss(variable)
        grad = torch.autograd.grad(lg, variable)
        y.append(grad[0])
        if i==fprim:
            plt.scatter(i, grad[0], color=color)


    plt.plot(x, y, label=loss, color=color)


plt.xlabel('Value of V(x)')
plt.ylabel('Gradient')
plt.legend()
axes = plt.gca()
# axes.set_xlim([xmin,xmax])
axes.set_ylim([-10,10])
plt.title('Gradient of the generator vs V(x)')
plt.savefig(f"Grad_Graphs/GradientofGeneratorVsVx.png")
plt.close(fig)



x=[i/10 for i in range (-50, 50, 1)]
#Discriminator
fig = plt.figure()
plt.ioff()
# zip(['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon'],['blue','orange','purple','green','pink','gray'])
for loss,color,fprim in zip(['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon'],['blue','orange','purple','green','pink','gray'], fprim1):
    y = []
    div = Divergence(loss)
    variable = torch.zeros(1, requires_grad=True)
    for i in x:
        arr=[]
        for j in x:
            variable = torch.zeros(1, requires_grad=True) + i
            variable2 = torch.zeros(1, requires_grad=True)+j
            lg = div.D_loss(DX_score=variable, DG_score=variable2)
            grad = torch.autograd.grad(lg, variable)
            grad2=torch.autograd.grad(lg, variable2)
            arr.append(grad[0]+grad2[0])
        y.append(arr)

    figure, axes = plt.subplots()
    plt.imshow(y, cmap='hot', interpolation='nearest')

    # plt.plot(x, y, label=loss, color=color)


    plt.xlabel('Value of V(x) from the real distribution')
    plt.ylabel('Value of V(x) from the generated distribution')
    plt.legend()
    axes = plt.gca()
    plt.colorbar()
    # axes.axis([-5,5,-5,5])
    # axes.set_xlim([xmin,xmax])
    # axes.set_ylim([-10,10])
    plt.xticks(np.arange(0,100, 10), [-5,-4,-3,-2,-1,0,1,2,3,4])
    plt.yticks(np.arange(0,100, 10), [-5,-4,-3,-2,-1,0,1,2,3,4])
    plt.title(f'Gradient of the discriminator vs V(x) for {loss} divergence')
    plt.savefig(f"Grad_Graphs/GradientOfDiscriminator{loss}.png")
    plt.close(fig)
