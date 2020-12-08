import torch
from torch import optim
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import get_data, visualize_tsne, plot_losses, estimate_mu_var, get_data_q
from f_gan import Generator, Critic, Divergence
import argparse
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import ast
torch.manual_seed(3435)
#MAYBE IT DOESNT LEARN BECAUSE VANISHING GRADIENT
#TODO: PLOT
#TODO: SEE IF VANISHING GRADIENT
#TOTO COMPARE WITH FULL LEARNING

def run_exp(argsdict):
    # Example of usage of the code provided and recommended hyper parameters for training GANs.
    n_iter = argsdict['nb_epoch'] # N training iterations
    lr = 3e-4
    z_dim = argsdict['Gauss_size']
    hidden_dim=(400, 400)

    image_shape=(1, 28, 28)
    encoding='sigmoid'
    # Finding random mu
    mus = []
    sigma=[]
    for gaus in range(argsdict['number_gaussians']):
        mus.append([random.randint(-1, -1) for _ in range(argsdict['Gauss_size'])])
        sigma.append([random.randint(1, 1) for _ in range(argsdict['Gauss_size'])])
    mus = torch.tensor(mus)
    sigma=torch.tensor(sigma)

    mus2 = []
    sigma2 = []
    for gaus in range(argsdict['number_gaussians']):
        mus2.append([random.randint(1, 1) for _ in range(argsdict['Gauss_size'])])
        sigma2.append([random.randint(4, 4) for _ in range(argsdict['Gauss_size'])])
    mus2 = torch.tensor(mus2)
    sigma2 = torch.tensor(sigma2)
    argsdict['mus'] = mus
    argsdict['sigma']=sigma
    argsdict['musq'] = mus2
    argsdict['sigmaq'] = sigma2

    # Use the GPU if you have one
    if torch.cuda.is_available():
        print("Using the GPU")
        device = torch.device("cuda")
    else:
        print("WARNING: You are about to run on cpu, and this will likely run out \
          of memory. \n You can try setting batch_size=1 to reduce memory usage")
        device = torch.device("cpu")

    train_loader, valid_loader, test_loader= get_data(argsdict)
    train_loaderq, valid_loader, test_loader = get_data_q(argsdict)
    print(device)
    generator = Generator(image_shape, hidden_dim[0], hidden_dim[1], z_dim, encoding, argsdict).to(device)
    critic = Critic(argsdict['Gauss_size'], argsdict['crit_size'], argsdict['crit_size']).to(device)

    optim_critic = optim.Adam(critic.parameters(), lr=lr)#, betas=(beta1, beta2))
    optim_generator = optim.SGD(generator.parameters(), lr=lr)#, betas=(beta1, beta2))

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
        for i_batch, (sample_batch, sample_batch_q) in enumerate(zip(train_loader, train_loaderq)):
            train_batch_size = argsdict['batch_size']
            noise = Variable(
                torch.normal(torch.zeros(train_batch_size, z_dim), torch.ones(train_batch_size, z_dim))).cuda()
            optim_critic.zero_grad()
            if argsdict['use_cuda']:
                real_img, label_batch=sample_batch[0].cuda(), sample_batch[1]
            else:
                real_img, label_batch=sample_batch[0], sample_batch[1]
            fake_img=sample_batch_q[0]
            if argsdict['train_generator']:
                fake_img = generator(noise)
            #Attempting loss
            DX_score=critic(real_img)
            DG_score=critic(fake_img)
            loss_D=losses.D_loss(DX_score, DG_score)
            # print(loss_D)
            # muq, varq = estimate_mu_var(fake_img.view(-1, image_shape[1], image_shape[2]))
            # fdiv = losses.AnalyticDiv(argsdict['musq'], argsdict['mus'], argsdict['sigmaq'], argsdict['sigma'])
            D_losses.append(loss_D.item())

            optim_generator.zero_grad()
            if argsdict['train_generator'] and i_batch%1==0:
                noise = Variable(
                    torch.normal(torch.zeros(train_batch_size, z_dim), torch.ones(train_batch_size, z_dim))).cuda()
                gen_img=generator(noise)
                # print(gen_img.shape)
                DG_score=critic(gen_img)
                loss_G = losses.G_loss(DG_score)
                # print(loss_G.item())
                G_losses.append(loss_G.item())
                loss_G.backward()
                # print("HELLO")
                # print(generator.mu.grad)
                # print(generator.mu)
                optim_generator.step()
                torch.clip(generator.sigma, 0)


            # print(D_losses)
            fake, real=losses.RealFake(DG_score, DX_score)
            real_stat.append(real)
            fake_stat.append(fake)
            loss_D.backward()
            optim_critic.step()

        print("Epoch[%d/%d], G Loss: %.4f  D Loss: %.4f"
              % (epoch, n_iter,np.mean(G_losses), np.mean(D_losses)))
        print(f"Classified on average {round(np.mean(real_stat), 2)} real examples correctly and {round(np.mean(fake_stat), 2)} fake examples correctly")
        print(f"real mu: {argsdict['mus']}, real sigma: {argsdict['sigma']}")
        # muq, varq=estimate_mu_var(img.view(-1, image_shape[1], image_shape[2]))
        if argsdict['train_generator']:
            print(f"generated mu: {generator.mu}, generated sigma: {generator.sigma}")
            print(f"f divergence {losses.AnalyticDiv(generator.mu.unsqueeze(0), argsdict['mus'], generator.sigma.unsqueeze(0), argsdict['sigma'])}")
        else:
            #TODO SOMEWHERE RANDOM IS RESET EACH EPOCH
            print(f"generated mu: {argsdict['musq']}, generated sigma: {argsdict['sigmaq']}")
            print(f"f divergence {losses.AnalyticDiv(argsdict['musq'], argsdict['mus'], argsdict['sigmaq'], argsdict['sigma'])}")
            muEst=torch.mean(real_img, dim=0).unsqueeze(0)
            stdEst=torch.var(real_img, unbiased=True, dim=0).unsqueeze(0)
            muhatEst=torch.mean(fake_img, dim=0).unsqueeze(0)
            stdhatEst=torch.var(fake_img, unbiased=True, dim=0).unsqueeze(0)
            print(f"Real f divergence {losses.AnalyticDiv(muhatEst, muEst, stdhatEst, stdEst)}")

        losses_Generator.append(np.mean(G_losses))
        losses_Discriminator.append(np.mean(D_losses))
        real_statistics.append(np.mean(real_stat))
        fake_statistics.append(np.mean(fake_stat))
        if argsdict['dataset']=='Gaussian':
            #A bit hacky but reset iterators
            train_loader, valid_loader, test_loader = get_data(argsdict)
            train_loaderq, valid_loader, test_loader = get_data_q(argsdict)
        with torch.no_grad():
            img=generator(Fix_Noise)

        # #Saving Images
        # if argsdict['modified_loss']:
        #     save_image(img.view(-1, image_shape[0], image_shape[1], image_shape[2]), f"{argsdict['dataset']}_IMGS/{argsdict['divergence']}/GRID_trick32%d.png" % epoch, nrow=5, normalize=True)
        # else:
        #     save_image(img.view(-1, image_shape[0], image_shape[1], image_shape[2]),f"{argsdict['dataset']}_IMGS/{argsdict['divergence']}/GRID%d.png" % epoch, nrow=5,normalize=True)
        with open(f"{argsdict['dataset']}_IMGS/{argsdict['divergence']}/Losses.txt", "w") as f:
            json.dump({'Gen_Loss':losses_Generator, 'Discri_Loss':losses_Discriminator, 'real_stat':real_statistics, 'fake_stat':fake_statistics}, f)
    
        # Update the losses plot every 5 epochs
        # if epoch%5==0 and epoch!=0:
        #     plot_losses(argsdict, epoch+1, show_plot=0)
                  
    # plot_losses(argsdict, n_iter)

    return -np.mean(D_losses), losses.AnalyticDiv(argsdict['musq'], argsdict['mus'], argsdict['sigmaq'], argsdict['sigma']), losses.AnalyticDiv(muhatEst, muEst, stdhatEst, stdEst)

#BS: 500: 0.4703 vs 0.44315
#BS: 5000
#I need to train it before so that's its actually a gaussian, and then try with different batch size

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Project for IFT6269 on fgans')
    parser.add_argument('--dataset', type=str, default='Gaussian',
                        help='Dataset you want to use. Options include MNIST, svhn, Gaussian, and CIFAR')
    parser.add_argument('--divergence', type=str, default='forward_kl',
                        help='divergence to use. Options include total_variation, forward_kl, reverse_kl, pearson, hellinger, jensen_shannon, alpha_div or all')
    parser.add_argument('--Gauss_size', type=int, default=1, help='The size of the Gaussian we generate')
    parser.add_argument('--number_gaussians', type=int, default='1', help='The number of Gaussian we generate')
    parser.add_argument('--dataset_size', type=int, default=50000, help='the total number of points generated by the gaussian')
    parser.add_argument('--num_gen', type=int, default=5, help='number of point generated by both the dataset and generator')
    parser.add_argument('--threshold', type=float, default=0.5,help='threshold after which the data point is considered as part of the generated distribution')
    #Training options
    parser.add_argument('--nb_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=500, help='batch size for training and testing')
    parser.add_argument('--modified_loss', action='store_true', help='use the loss of section 3.2 instead of the original formulation')
    parser.add_argument('--test_capacity', action='store_true', help='Test the lower bound vs capacity')
    parser.add_argument('--test_dimensions', action='store_true', help='Test the lower bound vs dimensions')
    parser.add_argument('--crit_size', type=int, default=32)
    parser.add_argument('--visualize', action='store_true', help='save visualization of the datasets using t-sne')
    parser.add_argument('--use_cuda', action='store_true', help='Use gpu')
    parser.add_argument('--fix_seed', action='store_true', help='Fix the seed')
    parser.add_argument('--train_generator', action='store_true', help='train the generator to match the distribution')
    args = parser.parse_args()

    argsdict = args.__dict__
    if argsdict['divergence']=='all':
        divergence=['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon','alpha_div']
        for dd in divergence:
            print(dd)
            argsdict['divergence']=dd
            run_exp(argsdict)
    elif argsdict['train_generator']:
        run_exp(argsdict)
    elif argsdict['test_dimensions']:
        estimated=[]
        true=[]
        sampled=[]
        arr=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for num_dim in arr:
            argsdict['batch_size']=1000
            argsdict['nb_epoch']=25
            argsdict['dataset_size']=argsdict['batch_size']*20
            argsdict['Gauss_size']=num_dim
            print(num_dim)
            Estimated, trueDiv, sampledDiv=run_exp(argsdict)
            estimated.append(Estimated.item())
            true.append(trueDiv.item())
            sampled.append(sampledDiv.item())
            with open(f"{argsdict['dataset']}_IMGS/{argsdict['divergence']}/LowerBoundVsDim.txt", "w") as f:
                    json.dump({"Estimated":estimated, "True":true, "Sampled":sampled, "num_dim":arr}, f)
    elif argsdict['test_capacity']:
        estimated=[]
        true=[]
        sampled=[]
        arr=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
        for crit_size in arr:
            argsdict['batch_size']=1000
            argsdict['nb_epoch']=25
            argsdict['dataset_size']=argsdict['batch_size']*20
            argsdict['Gauss_size']=10
            argsdict['crit_size']=crit_size
            print(crit_size)
            Estimated, trueDiv, sampledDiv=run_exp(argsdict)
            estimated.append(Estimated.item())
            true.append(trueDiv.item())
            sampled.append(sampledDiv.item())
            with open(f"{argsdict['dataset']}_IMGS/{argsdict['divergence']}/LowerBoundVsCapacity.txt", "w") as f:
                    json.dump({"Estimated":estimated, "True":true, "Sampled":sampled, "crit_size":arr}, f)