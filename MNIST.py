import torch
from torch import optim
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import get_data
from f_gan import Generator, Discriminator, Divergence
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Project for IFT6269 on fgans')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='Dataset you want to use. Options include MNIST, CelebA, and CIFAR')
    parser.add_argument('--divergence', type=str, default='jensen_shannon',
                        help='divergence to use. Options include total_variation, forward_kl, reverse_kl, pearson, hellinger, jensen_shannon')
    parser.add_argument('--batch_size', type=int, default='100', help='batch size for training and testing')
    args = parser.parse_args()

    argsdict = args.__dict__
    # Example of usage of the code provided and recommended hyper parameters for training GANs.
    data_root = './'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_iter = 50 # N training iterations
    n_critic_updates = 1 # N critic updates per generator update
    train_batch_size = 100
    test_batch_size = 100
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.9
    z_dim = 25
    hidden_dim=100

    # Use the GPU if you have one
    if torch.cuda.is_available():
        print("Using the GPU")
        device = torch.device("cuda")
    else:
        print("WARNING: You are about to run on cpu, and this will likely run out \
          of memory. \n You can try setting batch_size=1 to reduce memory usage")
        device = torch.device("cpu")

    train_loader, valid_loader, test_loader = get_data(argsdict)

    generator = Generator(784, hidden_dim, z_dim).to(device)
    critic = Discriminator(784, hidden_dim).to(device)

    #TODO Adding beta seems to make total variation go to 0, why.
    #TODO In rapport talk about how finicky the whole system is
    optim_critic = optim.Adam(critic.parameters(), lr=lr)#, betas=(beta1, beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=lr)#, betas=(beta1, beta2))

    losses=Divergence(argsdict['divergence'])

    Fix_Noise=Variable(torch.normal(torch.zeros(25, z_dim), torch.ones(25, z_dim))).cuda()
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
            optim_critic.step()

            #This helps the discriminator but it makes it learn way too fast and the generator cant learn
            # Clip weights of discriminator
            # for p in critic.parameters():
            #     p.data.clamp_(-0.01, 0.01)

            #train the generator ever n_critic iterations
            D_losses.append(loss_D.item())
            if i_batch %n_critic_updates==0:
                optim_generator.zero_grad()

                gen_imgs=generator(noise)
                DG_score=critic(gen_imgs)
                loss_G=losses.G_loss(DG_score)
                loss_G.backward()
                optim_generator.step()

            G_losses.append(loss_G.item())
        # print(G_losses)
        # print(D_losses)
        print("Epoch[%d/%d], G Loss: %.4f, D Loss: %.4f"
              % (epoch, n_iter, np.mean(G_losses), np.mean(D_losses)))
        with torch.no_grad():
            img=generator(Fix_Noise)
        # save_image(img.view(1, 28, 28), f"MNIST_IMGS/{argsdict['divergence']}/%d.png" % epoch)
        save_image(img.view(-1, 1, 28, 28), f"MNIST_IMGS/{argsdict['divergence']}/GRID%d.png" % epoch, nrow=5)