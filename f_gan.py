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
        self.image_size=image_size
        x = [nn.Linear(z_dim, hidden_dim2),
             nn.BatchNorm1d(hidden_dim2),
             nn.ReLU(inplace=True),
             # nn.Linear(hidden_dim, hidden_dim2),
             # nn.BatchNorm1d(hidden_dim2),
             # nn.ReLU(inplace=True),
             nn.Linear(hidden_dim2, image_size[0]*image_size[1]*image_size[2])]

        self.x = nn.Sequential(*x)
        self.encoding=encoding

    def forward(self, x):
        x=to_cuda(x.view(x.shape[0],-1))
        x=self.x(x)
        if self.encoding=='tanh':
            x = torch.tanh(x)
        elif self.encoding=='sigmoid':
            x = torch.sigmoid(x)
            # x=torch.round(x)
        x.reshape(x.shape[0], self.image_size[0],self.image_size[1],self.image_size[2])
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
        x = [nn.Linear(image_size[0]*image_size[1]*image_size[2], hidden_dim),
             nn.ELU(inplace=True),
             nn.Linear(hidden_dim, hidden_dim2),
             nn.ELU(inplace=True),
             nn.Linear(hidden_dim2, 1)]
        #TODO: I'm very unsure as to wether we should have a sigmoid at the end of the critic. The
        #OG implementation had one but the paper says "The final activation function is determined by the divergence"
        #So to check.

        self.x = nn.Sequential(*x)


    def forward(self, x):
        x = to_cuda(x.view(x.shape[0], -1))
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

        sigq=to_cuda(sigq.detach())
        sigp=to_cuda(sigp.detach())
        mup=to_cuda(mup.detach())
        muq=to_cuda(muq.detach())
        dim=mup.shape[1]
        sigq, muq, sigp, mup=sigq.unsqueeze(0), muq.unsqueeze(0), sigp, mup
        # print(mup.shape, muq.shape)
        # print(sigq.shape, sigp.shape)
        if self.method== "forward_kl":
            div=0
            div+=torch.sum(torch.log(sigq))
            # print(sigq)
            # print(div)
            div-=torch.sum(torch.log(sigp.float()))
            div+=torch.sum(sigp/sigq)
            # print(torch.matmul((muq-mup), torch.diag(sigq.view(dim))).shape)
            # print((muq-mup).T)
            div+=torch.matmul(torch.matmul((muq-mup), torch.diag(1/sigq.view(dim))), (muq-mup).T)[0, 0]
            div-=sigq.shape[1]
            print(0.5*div)
            return 0.5*div


class fGANTrainer:
    """ Object to hold data iterators, train a GAN variant
    """
    def __init__(self, model, train_iter, val_iter, test_iter, viz=False):
        self.model = to_cuda(model)
        self.name = model.__class__.__name__

        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter

        self.Glosses = []
        self.Dlosses = []

        self.viz = viz
        self.num_epochs = 0

    def train(self, num_epochs, method, G_lr=1e-4, D_lr=1e-4, D_steps=1):
        """ Train a standard vanilla GAN architecture using f-divergence as loss

            Logs progress using G loss, D loss, G(x), D(G(x)), visualizations
            of Generator output.

        Inputs:
            num_epochs: int, number of epochs to train for
            method: str, divergence metric to optimize
            G_lr: float, learning rate for generator's Adam optimizer
            D_lr: float, learning rate for discriminsator's Adam optimizer
            D_steps: int, ratio for how often to train D compared to G
        """
        # Initialize loss, indicate which GAN it is
        self.loss_fnc = Divergence(method)

        # Initialize optimizers
        G_optimizer = optim.Adam(params=[p for p in self.model.G.parameters()
                                        if p.requires_grad], lr=G_lr)
        D_optimizer = optim.Adam(params=[p for p in self.model.D.parameters()
                                        if p.requires_grad], lr=D_lr)

        # Approximate steps/epoch given D_steps per epoch
        # --> roughly train in the same way as if D_step (1) == G_step (1)
        epoch_steps = int(np.ceil(len(self.train_iter) / (D_steps)))

        # Begin training
        for epoch in tqdm(range(1, num_epochs+1)):

            self.model.train()
            G_losses, D_losses = [], []

            for _ in range(epoch_steps):

                D_step_loss = []

                for _ in range(D_steps):

                    # Reshape images
                    images = self.process_batch(self.train_iter)

                    # TRAINING D: Zero out gradients for D
                    D_optimizer.zero_grad()

                    # Train D to discriminate between real and generated images
                    D_loss = self.train_D(images)

                    # Update parameters
                    D_loss.backward()
                    D_optimizer.step()

                    # Log results, backpropagate the discriminator network
                    D_step_loss.append(D_loss.item())

                # So that G_loss and D_loss have the same number of entries.
                D_losses.append(np.mean(D_step_loss))

                # TRAINING G: Zero out gradients for G
                G_optimizer.zero_grad()

                # Train G to generate images that fool the discriminator
                G_loss = self.train_G(images)

                # Log results, update parameters
                G_losses.append(G_loss.item())
                G_loss.backward()
                G_optimizer.step()

            # Save progress
            self.Glosses.extend(G_losses)
            self.Dlosses.extend(D_losses)

            # Progress logging
            print ("Epoch[%d/%d], G Loss: %.4f, D Loss: %.4f"
                   %(epoch, num_epochs, np.mean(G_losses), np.mean(D_losses)))
            self.num_epochs += 1

            # Visualize generator progress
            if self.viz:
                self.generate_images(epoch)
                plt.show()

    def train_D(self, images):
        """ Run 1 step of training for discriminator

        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            D_loss: f-divergence between generated, true distributions
        """
        # Classify the real batch images, get the loss for these
        DX_score = self.model.D(images)

        # Sample noise z, generate output G(z)
        noise = self.compute_noise(images.shape[0], self.model.z_dim)
        G_output = self.model.G(noise)

        # Classify the fake batch images, get the loss for these using sigmoid cross entropy
        DG_score = self.model.D(G_output)

        # Compute f-divergence loss
        D_loss = self.loss_fnc.D_loss(DX_score, DG_score)

        return D_loss

    def train_G(self, images):
        """ Run 1 step of training for generator

        Input:
            images: batch of images reshaped to [batch_size, -1]
        Output:
            G_loss: f-divergence for difference between generated, true distributiones
        """
        # Get noise (denoted z), classify it using G, then classify the output
        # of G using D.
        noise = self.compute_noise(images.shape[0], self.model.z_dim) # z
        G_output = self.model.G(noise) # G(z)
        DG_score = self.model.D(G_output) # D(G(z))

        # Compute f-divergence loss
        G_loss = self.loss_fnc.G_loss(DG_score)

        return G_loss

    def compute_noise(self, batch_size, z_dim):
        """ Compute random noise for input into Generator G """
        return to_cuda(torch.randn(batch_size, z_dim))

    def process_batch(self, iterator):
        """ Generate a process batch to be input into the Discriminator D """
        images, _ = next(iter(iterator))
        return images

    def generate_images(self, epoch, num_outputs=36, save=True):
        """ Visualize progress of generator learning """
        # Turn off any regularization
        self.model.eval()

        # Sample noise vector
        noise = self.compute_noise(num_outputs, self.model.z_dim)

        # Transform noise to image
        images = self.model.G(noise)

        # Reshape to proper image size
        images = images.view(images.shape[0],
                             self.model.shape,
                             self.model.shape,
                             -1).squeeze()

        # Plot
        plt.close()
        grid_size, k = int(num_outputs**0.5), 0
        fig, ax = plt.subplots(grid_size, grid_size, figsize=(5, 5))
        for i, j in product(range(grid_size), range(grid_size)):
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
            images=images.cpu()
            ax[i,j].imshow(images[k].data.cpu().numpy(), cmap='gray')
            k += 1

        # Save images if desired
        if save:
            outname = '../viz/' + self.name + '/' + self.loss_fnc.method + '/'
            if not os.path.exists(outname):
                os.makedirs(outname)
            torchvision.utils.save_image(images.unsqueeze(1).data,
                                         outname + 'reconst_%d.png'
                                         %(epoch), nrow=grid_size)

    def viz_loss(self):
        """ Visualize loss for the generator, discriminator """
        # Set style, figure size
        plt.style.use('ggplot')
        plt.rcParams["figure.figsize"] = (8,6)

        # Plot Discriminator loss in red
        plt.plot(np.linspace(1, self.num_epochs, len(self.Dlosses)),
                 self.Dlosses,
                 'r')

        # Plot Generator loss in green
        plt.plot(np.linspace(1, self.num_epochs, len(self.Dlosses)),
                 self.Glosses,
                 'g')

        # Add legend, title
        plt.legend(['Discriminator', 'Generator'])
        plt.title(self.name + ' : ' + self.loss_fnc.method)
        plt.show()

    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath)

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)


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
