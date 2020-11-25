import torch
from torch import optim
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import get_data, visualize_tsne
from f_gan import Generator, Critic, Divergence
import argparse
import numpy as np
import json



def run_exp(argsdict):
    # Example of usage of the code provided and recommended hyper parameters for training GANs.
    data_root = './'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_iter = 50 # N training iterations
    n_critic_updates = 1 # N critic updates per generator update
    train_batch_size = 64
    test_batch_size = 512
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.9
    z_dim = 25
    hidden_dim=(64, 16)

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
        image_shape=(1, 1, argsdict['Gauss_size'])
        encoding='tanh'
    # elif argsdict['dataset'] in ['Gaussian']:
    #

    # Use the GPU if you have one
    if torch.cuda.is_available():
        print("Using the GPU")
        device = torch.device("cuda")
    else:
        print("WARNING: You are about to run on cpu, and this will likely run out \
          of memory. \n You can try setting batch_size=1 to reduce memory usage")
        device = torch.device("cpu")

    train_loader, valid_loader, test_loader = get_data(argsdict)
    print(device)
    generator = Generator(image_shape, hidden_dim[0], hidden_dim[1], z_dim, encoding).to(device)
    # generator = Generatorsvhn(z_dim, hidden_dim).to(device)
    critic = Critic(image_shape, 32, 32).to(device)
    # critic = Criticsvhn(argsdict['hidden_discri_size']).to(device)

    #TODO Adding beta seems to make total variation go to 0, why.
    #TODO In rapport talk about how finicky the whole system is
    optim_critic = optim.Adam(critic.parameters(), lr=lr)#, betas=(beta1, beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=lr)#, betas=(beta1, beta2))

    losses=Divergence(argsdict['divergence'])

    Fix_Noise=Variable(torch.normal(torch.zeros(25, z_dim), torch.ones(25, z_dim))).cuda()

    losses_Generator=[]
    losses_Discriminator=[]

    # COMPLETE TRAINING PROCEDURE
    for epoch in range(n_iter):
        G_losses, D_losses=[], []
        for i_batch, sample_batch in enumerate(train_loader):
            optim_critic.zero_grad()
            real_img, label_batch=sample_batch[0].cuda(), sample_batch[1]
            #fake img
            noise=Variable(torch.normal(torch.zeros(train_batch_size, z_dim), torch.ones(train_batch_size, z_dim))).cuda()
            fake_img=generator(noise)
            #Attempting loss
            DX_score=critic(real_img)
            DG_score=critic(fake_img)
            loss_D=losses.D_loss(DX_score, DG_score)

            loss_D.backward()
            # D_grad=critic.x[0].weight.grad.detach()
            optim_critic.step()


            #train the generator ever n_critic iterations
            D_losses.append(loss_D.item())
            if i_batch %n_critic_updates==0:
                optim_generator.zero_grad()

                gen_imgs=generator(noise)
                DG_score=critic(gen_imgs)
                # loss_G=losses.G_loss(DG_score)
                loss_G = losses.G_loss(DG_score)
                loss_G.backward()
                optim_generator.step()

            G_losses.append(loss_G.item())
        # print(G_losses)
        # print(D_losses)
        # print(D_grad)
        print("Epoch[%d/%d], G Loss: %.4f, D Loss: %.4f"
              % (epoch, n_iter, np.mean(G_losses), np.mean(D_losses)))
        losses_Generator.append(np.mean(G_losses))
        losses_Discriminator.append(np.mean(D_losses))
        if argsdict['dataset']=='Gaussian':
            #A bit hacky but reset iterators
            train_loader, valid_loader, test_loader = get_data(argsdict)
        if argsdict['visualize']:
            visualize_tsne(fake_img, real_img, argsdict, epoch)
        with torch.no_grad():
            img=generator(Fix_Noise)
        save_image(img.view(-1, image_shape[0], image_shape[1], image_shape[2]), f"{argsdict['dataset']}_IMGS/{argsdict['divergence']}/GRID%d.png" % epoch, nrow=5, normalize=True)
        with open(f"{argsdict['dataset']}_IMGS/{argsdict['divergence']}/Losses.txt", "w") as f:
            json.dump({'Gen_Loss':losses_Generator, 'Discri_Loss':losses_Discriminator}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Project for IFT6269 on fgans')
    parser.add_argument('--dataset', type=str, default='svhn',
                        help='Dataset you want to use. Options include MNIST, svhn, Gaussian, and CIFAR')
    parser.add_argument('--divergence', type=str, default='total_variation',
                        help='divergence to use. Options include total_variation, forward_kl, reverse_kl, pearson, hellinger, jensen_shannon, or all')
    parser.add_argument('--Gauss_size', type=int, default='30', help='The size of the Gaussian we generate')

    #Training options
    parser.add_argument('--batch_size', type=int, default='64', help='batch size for training and testing')
    parser.add_argument('--hidden_crit_size', type=int, default=32)

    parser.add_argument('--visualize', action='store_true', help='save visualization of the datasets using t-sne')
    args = parser.parse_args()

    argsdict = args.__dict__

    if argsdict['divergence']=='all':
        divergence=['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon']
        for dd in divergence:
            print(dd)
            argsdict['divergence']=dd
            run_exp(argsdict)
    else:
        run_exp(argsdict)