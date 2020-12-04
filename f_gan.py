""" (f-GAN) https://arxiv.org/abs/1606.00709
f-Divergence GANs

The authors empirically demonstrate that when the generative model is
misspecified and does not contain the true distribution, the divergence
function used for estimation has a strong influence on which model is
learned. To address this issue, they theoretically show that the
generative-adversarial approach is a special case of an existing, more
general variational divergence estimation approach and that any
f-divergence can be used for training generative neural samplers (which
are defined as models that take a random input vector and produce a sample
from a probability distribution defined by the network weights). They
then empirically show the effect of using different training
divergences on a trained model's average log likelihood of sampled data.

They test (forward) Kullback-Leibler, reverse Kullback-Leibler, Pearson
chi-squared, Neyman chi-squared, squared Hellinger, Jensen-Shannon,
and Jeffrey divergences.

We exclude Neyman and Jeffrey due to poor performance and nontrivial
implementations to yield 'convergence' (see scipy.special.lambertw
for how to implement Jeffrey, and Table 6 of Appendix C of the paper
for how to implement Neyman)
"""

import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import matplotlib.pyplot as plt
import numpy as np

from itertools import product
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from utils import *


# class Generator(nn.Module):
#     """ Generator. Input is noise, output is a generated image.
#     """
#     def __init__(self, image_size, hidden_dim, z_dim):
#         super().__init__()
#         self.linear = nn.Linear(z_dim, hidden_dim)
#         self.generate = nn.Linear(hidden_dim, image_size)
#
#     def forward(self, x):
#         x=x.view(x.shape[0],-1)
#         activated=F.relu(self.linear(x))
#         # generation=torch.sigmoid(activated)
#         generation = torch.sigmoid(self.generate(activated))
#         generation.reshape(generation.shape[0], 1, 28, 28)
#         return generation


# class Discriminator(nn.Module):
#     """ Discriminator. Input is an image (real or generated),
#     output is P(generated).
#     """
#     def __init__(self, image_size, hidden_dim):
#         super().__init__()
#         self.linear = nn.Linear(image_size, hidden_dim)
#         self.discriminate = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x):
#         x = to_cuda(x.view(x.shape[0], -1))
#         print(x.shape)
#         activated = F.relu(self.linear(x))
#         discrimination = torch.sigmoid(self.discriminate(activated))
#         return discrimination

#TODO Generate with these for MNIST
#
# class Generator(nn.Module):
#     """ Generator. Input is noise, output is a generated image.
#     """
#     def __init__(self, image_size, hidden_dim, hidden_dim2, z_dim, encoding):
#         super().__init__()
#         self.image_size=image_size
#         x = [nn.Linear(z_dim, hidden_dim),
#              nn.BatchNorm1d(hidden_dim),
#              nn.ReLU(inplace=True),
#              nn.Linear(hidden_dim, hidden_dim2),
#              nn.BatchNorm1d(hidden_dim2),
#              nn.ReLU(inplace=True),
#              nn.Linear(hidden_dim2, image_size[0]*image_size[1]*image_size[2])]
#
#         self.x = nn.Sequential(*x)
#         self.encoding=encoding
#
#     def forward(self, x):
#         x=to_cuda(x.view(x.shape[0],-1))
#         x=self.x(x)
#         if self.encoding=='tanh':
#             x = torch.tanh(x)
#         elif self.encoding=='sigmoid':
#             x = torch.sigmoid(x)
#         x.reshape(x.shape[0], self.image_size[0],self.image_size[1],self.image_size[2])
#         return x


class Generator(nn.Module):
    """ Generator. Input is noise, output is a generated image.
    """
    def __init__(self, image_size, hidden_dim, hidden_dim2, z_dim, encoding, argsdict):
        super().__init__()
        gauss_dim=argsdict['Gauss_size']
        self.image_size=image_size
        self.mu=nn.Parameter(torch.randn(gauss_dim))
        self.sigma=nn.Parameter(torch.abs(torch.randn(gauss_dim)))
        # self.param=nn.ParameterList(self.mu, self.sigma)

    def forward(self, x):
        # print(x.shape)
        x=to_cuda(x.view(x.shape[0],-1))
        # print(x.shape)
        # print(self.sigma.unsqueeze(0).repeat(x.shape[0], 1).shape)
        x=x*self.sigma.unsqueeze(0).repeat(x.shape[0], 1)+self.mu
        # x=x+self.mu
        # x.reshape(x.shape[0], self.image_size[0], -1)
        # print(x.shape)
        return x

#
# class Generator(nn.Module):
#     """ Generator. Input is noise, output is a generated image.
#     """
#     def __init__(self, image_size, hidden_dim, hidden_dim2, z_dim, encoding, argsdict):
#         super().__init__()
#         self.gauss_dim=argsdict['Gauss_size']
#         self.num_gaus=argsdict['number_gaussians']
#         self.argsdict=argsdict
#         self.image_size=image_size
#         self.mu=nn.Parameter(torch.Tensor([random.randint(0, 27), random.randint(0, 27)]))
#         self.sigma=nn.Parameter(torch.Tensor([random.randint(1, 10), random.randint(1, 10)]))
#         print(self.mu, self.sigma)
#         # self.param=nn.ParameterList(self.mu, self.sigma)
#
#     def forward(self, x):
#         # print(x.shape)
#         x=to_cuda(x.view(x.shape[0],-1))
#         batch_size=x.shape[0]
#         # print(x.shape)
#         # print(self.sigma.unsqueeze(0).repeat(x.shape[0], 1).shape)
#         bb = torch.zeros((batch_size, 1, 28, 28))
#         # Choose random gaussian
#         for j in range(batch_size):
#             grid = torch.zeros(1, 28, 28)
#             for k in range(self.argsdict['num_gen']):
#                 gaus = random.randint(0, self.argsdict['number_gaussians'] - 1)
#                 # mu = self.mu
#                 # sigma = self.sigma
#                 point = torch.round(self.sigma * torch.randn(self.argsdict['Gauss_size']).cuda() + self.mu)
#                 # print(point)
#                 point = torch.clip(point, 0, 27)
#                 grid[0, int(point[0]), int(point[1])] = 1
#             bb[j] = grid
#         return bb, self.mu, self.sigma

#TODO: Batch size vs lower bound
#TODO:

#
class Critic(nn.Module):
    """ Discriminator. Input is an image (real or generated),
    output is P(generated).
    """
    def __init__(self, image_size, hidden_dim, hidden_dim2):
        super().__init__()
        self.image_size = image_size
        x = [nn.Linear(image_size, hidden_dim),
             nn.ReLU(),
             nn.Linear(hidden_dim, hidden_dim2),
             nn.ReLU(),
             nn.Linear(hidden_dim2, 1)]
        #TODO: I'm very unsure as to wether we should have a sigmoid at the end of the critic. The
        #OG implementation had one but the paper says "The final activation function is determined by the divergence"
        #So to check.

        self.x = nn.Sequential(*x)


    def forward(self, x):
        x = to_cuda(x.view(x.shape[0], -1))
        # print(x[0])
        x = self.x(x)
        return x

class fGAN(nn.Module):
    """ Super class to contain both Discriminator (D) and Generator (G)
    """
    def __init__(self, image_size, hidden_dim, z_dim):
        super().__init__()

        self.__dict__.update(locals())

        self.G = Generator(image_size, hidden_dim, z_dim)
        self.D = Discriminator(image_size, hidden_dim)

        self.shape = int(image_size ** 0.5)


class Divergence:
    """ Compute G and D loss using an f-divergence metric.
    Implementations based on Table 6 (Appendix C) of the arxiv paper.
    """
    def __init__(self, method):
        self.method = method.lower().strip()
        assert self.method in ['total_variation',
                               'forward_kl',
                               'reverse_kl',
                               'pearson',
                               'alpha_div',
                               'hellinger',
                               'jensen_shannon'], \
            'Invalid divergence.'
        if method not in ['total_variation',
                               'forward_kl',
                               'reverse_kl',
                               'pearson',
                               'hellinger',
                               'jensen_shannon']:
            print("Some functions have not been implemented correctly for this divergence and so results may not be reliable. I hope you know what you're doing")

    def D_loss(self, DX_score, DG_score):
        """ Compute batch loss for discriminator using f-divergence metric """
        #This formula can also be used to calculated the total f-divergence
        if self.method == 'total_variation':
            return -(torch.mean(0.5*torch.tanh(DX_score)) \
                        - torch.mean(0.5*torch.tanh(DG_score)))

        elif self.method == 'forward_kl':
            return -(torch.mean(DX_score) - torch.mean(torch.exp(DG_score-1)))

        elif self.method == 'reverse_kl':
            return -(torch.mean(-torch.exp(DX_score)) - torch.mean(-1-DG_score))

        elif self.method == 'pearson':
            return -(torch.mean(DX_score) - torch.mean(0.25*DG_score**2 + DG_score))

        elif self.method == 'hellinger':
            return -(torch.mean(1-torch.exp(DX_score)) \
                        - torch.mean((1-torch.exp(DG_score))/(torch.exp(DG_score))))

        elif self.method == 'jensen_shannon':
            return -(torch.mean(torch.tensor(2.)-(1+torch.exp(-DX_score))) \
                        - torch.mean(-(torch.tensor(2.)-torch.exp(DG_score))))

        elif self.method == 'alpha_div':
            #for alpha >1 
            alpha = 1.5
            return -(torch.mean(DX_score)-torch.mean((1./alpha)*(DG_score*(alpha-1.) + 1.)**(alpha/(alpha-1)) -1./alpha ))

    def G_loss(self, DG_score):
        """ Compute batch loss for generator using f-divergence metric """

        if self.method == 'total_variation':
            return -torch.mean(0.5*torch.tanh(DG_score))

        elif self.method == 'forward_kl':
            return -torch.mean(torch.exp(DG_score-1))

        elif self.method == 'reverse_kl':
            return -torch.mean(-1-DG_score)

        elif self.method == 'pearson':
            return -torch.mean(0.25*DG_score**2 + DG_score)

        elif self.method == 'hellinger':
            return -torch.mean((1-torch.exp(DG_score))/(torch.exp(DG_score)))

        elif self.method == 'jensen_shannon':
            return -torch.mean(-(torch.tensor(2.)-torch.exp(DG_score)))

        elif self.method == 'alpha_div':
            # for alpha > 1
            alpha = 1.5
            return -torch.mean(1./alpha*(DG_score*(alpha-1.) + 1.)**(alpha/(alpha-1)) -1./alpha )

    #modifying the generator loss (trick 3.2) for         
    def G_loss_modified_sec_32(self, DG_score):
            """ Compute batch loss for generator using f-divergence metric """

            if self.method == 'total_variation':
                return -torch.mean(0.5*torch.tanh(DG_score))

            elif self.method == 'forward_kl':
                return -torch.mean(DG_score)

            elif self.method == 'reverse_kl':
                return -torch.mean(-torch.exp(DG_score))

            elif self.method == 'pearson':
                return -torch.mean(DG_score)

            elif self.method == 'hellinger':
                return -torch.mean(1-torch.exp(DG_score))

            elif self.method == 'jensen_shannon':
                return -torch.mean(torch.tensor(2.)-(1+torch.exp(-DG_score)))
            
            elif self.method == 'alpha_div':
                return -torch.mean(torch.tensor(2.)-(1+torch.exp(-DG_score)))

    #TODO CHANGE REAL FAKE THERES AN ERROR
    def RealFake(self, DG_score, DX_score):
        #Returns the percent of examples that were correctly classified by the discriminator
        if self.method == 'total_variation':
            thresh=0

        elif self.method == 'forward_kl':
            thresh=1

        elif self.method == 'reverse_kl':
            thresh=-1

        elif self.method == 'pearson':
            thresh=0

        elif self.method == 'hellinger':
            thresh=0

        elif self.method == 'jensen_shannon':
            thresh=0

        elif self.method == 'alpha_div':
            thresh=0
        #TODO In the paper its the inverse I think
        predGen = sum([1 if pred > thresh else 0 for pred in DG_score])
        predReal = sum([0 if pred > thresh else 1 for pred in DX_score])
        GenLen=DG_score.shape[0]
        RealLen=DX_score.shape[0]
        return float(predGen)/GenLen, float(predReal)/RealLen

    def AnalyticDiv(self, muq, mup, sigq, sigp, ):
        """Calculate the analytical divergence between p and q for a diagonal gaussian"""

        #TestTensor
        # muq=torch.ones(2)
        # mup=torch.ones(2).unsqueeze(0)
        # sigq=torch.ones(2)
        # sigp=torch.ones(2).unsqueeze(0)

        sigq=to_cuda(sigq.detach().float())
        sigp=to_cuda(sigp.detach().float())
        mup=to_cuda(mup.detach().float())
        muq=to_cuda(muq.detach().float())
        dim=mup.shape[1]
        # sigq, muq, sigp, mup=sigq, muq, sigp, mup
        # print(mup.shape, muq.shape)
        # print(sigq.shape, sigp.shape)
        if self.method== "forward_kl":
            div=0
            div+=torch.sum(torch.log(sigq))
            # print(sigq)
            # print(div)
            div-=torch.sum(torch.log(sigp.float()))
            div+=torch.sum(sigp*(1/sigq))
            # print(torch.matmul((muq-mup), torch.diag(sigq.view(dim))).shape)
            # print((muq-mup).T)
            div+=torch.matmul(torch.matmul((muq-mup), torch.diag(1/sigq.view(dim))), (muq-mup).T)[0, 0]
            div-=sigq.shape[1]
            return 0.5*div



if __name__ == '__main__':

    # Load in binarized MNIST data, separate into data loaders
    train_iter, val_iter, test_iter = get_data()

    # Init model
    model = fGAN(image_size=784,
                 hidden_dim=400,
                 z_dim=20)

    # Init trainer
    trainer = fGANTrainer(model=model,
                          train_iter=train_iter,
                          val_iter=val_iter,
                          test_iter=test_iter,
                          viz=False)
    # Train
    trainer.train(num_epochs=25,
                  method='jensen_shannon',
                  G_lr=1e-4,
                  D_lr=1e-4,
                  D_steps=1)
