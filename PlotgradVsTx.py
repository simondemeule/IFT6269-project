"""Plot the value of the gradient of the discriminator and generator vs the value of T(x)"""
import torch
from f_gan import Divergence
import matplotlib.pyplot as plt
import numpy as np

fprim1=[0,1,0,0,0,0,0]

variable2=torch.zeros(1, requires_grad=True)

x=[i/10 for i in range (-50, 50, 1)]
y=[]

argdict={'trueDiv':'forward_kl', 'falseDiv':'forward_kl'}

fig = plt.figure()
plt.ioff()
# zip(['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon'],['blue','orange','purple','green','pink','gray'])
for loss,color,fprim in zip(['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon', 'piecewise'],['blue','orange','purple','green','pink','gray', 'black'], fprim1):
    y = []
    div = Divergence(loss, argdict)
    variable = torch.zeros(1, requires_grad=True)
    for i in x:
        variable = torch.zeros(1, requires_grad=True) + i
        # variable2 = torch.zeros(1, requires_grad=True)+j
        lg = div.G_loss(variable)
        # print(loss)
        # print(lg)
        grad = torch.autograd.grad(lg, variable)
        y.append(grad[0].cpu())
        if i==fprim and loss!='piecewise':
            plt.scatter(i, grad[0].cpu(), color=color)


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


fprim1=[0,1,0,0,0,0, 0]

x=[i/10 for i in range (-50, 50, 1)]
#Discriminator
fig = plt.figure()
plt.ioff()
# zip(['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon'],['blue','orange','purple','green','pink','gray'])
for loss,color,fprim in zip(['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon', 'piecewise'],['blue','orange','purple','green','pink','gray', 'black'], fprim1):
    print(loss)
    y = []
    div = Divergence(loss)
    variable = torch.zeros(1, requires_grad=True)
    #Fill Y axis from 5 to -5
    for i in x[::-1]:
        arr=[]
        #Fill x axis from -5 to 5
        for j in x:
            variable = torch.zeros(1, requires_grad=True) + i
            variable2 = torch.zeros(1, requires_grad=True)+j
            lg = div.D_loss(DX_score=variable, DG_score=variable2)
            grad = torch.autograd.grad(lg, variable)
            grad2=torch.autograd.grad(lg, variable2)
            arr.append(grad[0]+grad2[0])
        y.append(arr)
    # print(fprim)
    # print(torch.Tensor([fprim]))
    # fprim=div.Tx(torch.Tensor([fprim]))[0].item()
    # print(fprim)
    line=np.zeros_like(x)+(-fprim+5)*10
    # print(line)
    figure, axes = plt.subplots()
    plt.imshow(y, cmap='hot', interpolation='nearest')
    plt.plot(np.arange(0,100), line, color='white')
    line = np.zeros_like(x) + (fprim + 5) * 10
    plt.plot(np.zeros_like(x)+(fprim+5)*10,np.arange(0,100), color='white')

    # plt.plot(x, y, label=loss, color=color)


    plt.xlabel('Value of V(x) from the real distribution')
    plt.ylabel('Value of V(x) from the generated distribution')
    # plt.legend()
    axes = plt.gca()
    plt.colorbar()
    # axes.axis([-5,5,-5,5])
    # axes.set_xlim([xmin,xmax])
    # axes.set_ylim([-10,10])
    # print(np.arange(0,100, 10))
    # print(np.append(np.arange(0,100, 10), 99))
    plt.xticks(np.append(np.arange(0,100, 10), 99), [-5,-4,-3,-2,-1,0,1,2,3,4, 5])
    plt.yticks(np.append(np.arange(0,100, 10), 99), [5,4,3,2,1,0,-1,-2,-3,-4, -5])
    if loss=='total_variation':
        title='Total Variation'
    elif loss=='forward_kl':
        title='Forward KL'
    elif loss=='reverse_kl':
        title='Reverse KL'
    elif loss=='pearson':
        title='Pearson'
    elif loss=='hellinger':
        title='Hellinger'
    elif loss=='jensen_shannon':
        title='Jensen Shannon'
    elif loss=='piecewise':
        title='Piecewise'
    plt.title(f'Gradient of the discriminator vs V(x) for the {title} divergence')
    plt.savefig(f"Grad_Graphs/GradientOfDiscriminator{loss}.png")
    plt.close(fig)
