import torch
from torch import optim
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import get_data, visualize_tsne, plot_losses
from f_gan import Generator, Critic, Divergence
import argparse
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import ast

def run_exp(argsdict):
    # Example of usage of the code provided and recommended hyper parameters for training GANs.
    data_root = './'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_iter = 30 # N training iterations
    n_critic_updates = 1 # N critic updates per generator update
    train_batch_size = argsdict['batch_size']
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.9
    z_dim = argsdict['Gauss_size']
    hidden_dim=(400, 400)

    if argsdict['dataset'] in ['svhn']:
        image_shape=(3, 32, 32)
        encoding='tanh'
    elif argsdict['dataset'] in ['CIFAR']:
        image_shape=(3, 32,32)
        encoding='sigmoid'
    elif argsdict['dataset'] in ['MNIST']:
        image_shape=(1, 28, 28)
        encoding='sigmoid'
    elif argsdict['dataset'] in ['Gaussian']:
        image_shape=(1, 28, 28)
        encoding='tanh'
        # Finding random mu
        mus = []
        sigma=[]
        for gaus in range(argsdict['number_gaussians']):
            mus.append([random.random() for _ in range(argsdict['Gauss_size'])])
            sigma.append([random.random() for _ in range(argsdict['Gauss_size'])])
        mus = torch.tensor(mus)
        sigma=torch.tensor(sigma)
        argsdict['mus'] = mus
        argsdict['sigma']=sigma

    # Use the GPU if you have one
    if torch.cuda.is_available():
        print("Using the GPU")
        device = torch.device("cuda")
    else:
        print("WARNING: You are about to run on cpu, and this will likely run out \
          of memory. \n You can try setting batch_size=1 to reduce memory usage")
        device = torch.device("cpu")

    train_loader, valid_loader, test_loader, num_samples = get_data(argsdict)
    print(device)
    print(num_samples)
    generator = Generator(image_shape, hidden_dim[0], hidden_dim[1], z_dim, encoding, argsdict['Gauss_size']).to(device)
    # generator = Generatorsvhn(z_dim, hidden_dim).to(device)
    critic = Critic(argsdict['Gauss_size'], 400, 400).to(device)
    # critic = Criticsvhn(argsdict['hidden_discri_size']).to(device)

    #TODO Adding beta seems to make total variation go to 0, why.
    #TODO In rapport talk about how finicky the whole system is
    optim_critic = optim.Adam(critic.parameters(), lr=lr)#, betas=(beta1, beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=lr)#, betas=(beta1, beta2))

    losses=Divergence(argsdict['divergence'])
    if argsdict['use_cuda']:
        Fix_Noise=Variable(torch.normal(torch.zeros(25, z_dim), torch.ones(25, z_dim))).cuda()
    else:
        Fix_Noise=Variable(torch.normal(torch.zeros(25, z_dim), torch.ones(25, z_dim)))

    losses_Generator=[]
    losses_Discriminator=[]
    real_statistics=[]
    fake_statistics=[]



    # COMPLETE TRAINING PROCEDURE
    for epoch in range(n_iter):
        G_losses, D_losses=[], []
        real_stat, fake_stat=[], []
        if argsdict['visualize']:
            real_imgs=torch.zeros([num_samples, image_shape[1], image_shape[2]])
        for i_batch, sample_batch in enumerate(train_loader):
            optim_critic.zero_grad()
            if argsdict['use_cuda']:
                real_img, label_batch=sample_batch[0].cuda(), sample_batch[1]
            else:
                real_img, label_batch=sample_batch[0], sample_batch[1]
            if argsdict['visualize']:
                real_imgs[i_batch*train_batch_size:i_batch*train_batch_size+train_batch_size]=real_img.squeeze(1)
            #fake img
            if argsdict['use_cuda']:
                noise=Variable(torch.normal(torch.zeros(train_batch_size, z_dim), torch.ones(train_batch_size, z_dim))).cuda()
            else:
                noise=Variable(torch.normal(torch.zeros(train_batch_size, z_dim), torch.ones(train_batch_size, z_dim)))
            fake_img, mu, sigma=generator(noise)
            #Attempting loss
            DX_score=critic(real_img)
            DG_score=critic(fake_img)
            loss_D=losses.D_loss(DX_score, DG_score)
            fake, real=losses.RealFake(DG_score, DX_score)
            real_stat.append(real)
            fake_stat.append(fake)
            loss_D.backward()
            # D_grad=critic.x[0].weight.grad.detach()
            optim_critic.step()

            # Clip weights of discriminator
            # for p in critic.parameters():
            #     p.data.clamp_(-0.1, 0.1)


            #train the generator ever n_critic iterations
            D_losses.append(loss_D.item())
            # if i_batch %n_critic_updates==0:
            #     optim_generator.zero_grad()
            #
            #     gen_img, mu, sigma=generator(noise)
            #     if argsdict['modified_loss']:
            #         DG_score = critic(gen_img)
            #         # We maximize instead of minimizing
            #         loss_G = losses.G_loss_modified_sec_32(DG_score)
            #     else:
            #         DG_score=critic(gen_img)
            #         loss_G = losses.G_loss(DG_score)
            #     loss_G.backward()
            #     optim_generator.step()
            #     torch.clip(generator.sigma, 0)
            #
            # G_losses.append(loss_G.item())
        # print(G_losses)
        # print(D_losses)
        # print(D_grad)
        print("Epoch[%d/%d], G Loss: %.4f, D Loss: %.4f"
              % (epoch, n_iter, np.mean(G_losses), np.mean(D_losses)))
        print(f"Classified on average {round(np.mean(real_stat), 2)} real examples correctly and {round(np.mean(fake_stat), 2)} fake examples correctly")
        print(argsdict['mus'], argsdict['sigma'])
        print(generator.mu, generator.sigma)
        losses.AnalyticDiv(generator.mu, argsdict['mus'], generator.sigma, argsdict['sigma'])
        losses_Generator.append(np.mean(G_losses))
        losses_Discriminator.append(np.mean(D_losses))
        real_statistics.append(np.mean(real_stat))
        fake_statistics.append(np.mean(fake_stat))
        if argsdict['dataset']=='Gaussian':
            #A bit hacky but reset iterators
            train_loader, valid_loader, test_loader, num_samples = get_data(argsdict)
        if argsdict['visualize']:
            if argsdict['use_cuda']:
                noise = Variable(torch.normal(torch.zeros(500, z_dim), torch.ones(500, z_dim))).cuda()
            else:
                noise = Variable(torch.normal(torch.zeros(500, z_dim), torch.ones(500, z_dim)))
            fake_imgs = generator(noise)
            visualize_tsne(fake_imgs, real_imgs[:500], argsdict, epoch)
        with torch.no_grad():
            img=generator(Fix_Noise)
        #Saving Images
        # if argsdict['modified_loss']:
        #     save_image(img.view(-1, image_shape[0], image_shape[1], image_shape[2]), f"{argsdict['dataset']}_IMGS/{argsdict['divergence']}/GRID_trick32%d.png" % epoch, nrow=5, normalize=True)
        # else:
        #     save_image(img.view(-1, image_shape[0], image_shape[1], image_shape[2]),f"{argsdict['dataset']}_IMGS/{argsdict['divergence']}/GRID%d.png" % epoch, nrow=5,normalize=True)
        # with open(f"{argsdict['dataset']}_IMGS/{argsdict['divergence']}/Losses.txt", "w") as f:
        #     json.dump({'Gen_Loss':losses_Generator, 'Discri_Loss':losses_Discriminator, 'real_stat':real_statistics, 'fake_stat':fake_statistics}, f)
    
        #Update the losses plot every 5 epochs
        # if epoch%5==0 and epoch!=0:
        #     plot_losses(argsdict, epoch+1, show_plot=0)
                  
    plot_losses(argsdict, n_iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Project for IFT6269 on fgans')
    parser.add_argument('--dataset', type=str, default='Gaussian',
                        help='Dataset you want to use. Options include MNIST, svhn, Gaussian, and CIFAR')
    parser.add_argument('--divergence', type=str, default='total_variation',
                        help='divergence to use. Options include total_variation, forward_kl, reverse_kl, pearson, hellinger, jensen_shannon, alpha_div or all')
    parser.add_argument('--Gauss_size', type=int, default='2', help='The size of the Gaussian we generate')
    parser.add_argument('--number_gaussians', type=int, default='1', help='The number of Gaussian we generate')

    #Training options
    parser.add_argument('--batch_size', type=int, default='64', help='batch size for training and testing')
    parser.add_argument('--modified_loss', action='store_true', help='use the loss of section 3.2 instead of the original formulation')
    parser.add_argument('--hidden_crit_size', type=int, default=32)
    parser.add_argument('--visualize', action='store_true', help='save visualization of the datasets using t-sne')
    parser.add_argument('--use_cuda', action='store_true', help='Use gpu')
    args = parser.parse_args()

    argsdict = args.__dict__
    if argsdict['divergence']=='all':
        divergence=['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon','alpha_div']
        for dd in divergence:
            print(dd)
            argsdict['divergence']=dd
            run_exp(argsdict)
    else:
        run_exp(argsdict)