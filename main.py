import torch
from torch import optim
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import get_data
from f_gan import Generator, Critic, Divergence
import argparse
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import ast
from plotting import plot_divergence_training, plot_divergence_other, plot_real_fake_training, visualize_tsne
from single_step_SGD import *

# utility data structure for storing divergence data
class DivergenceData:
    def __init__(self, name):
        self.name = name
        self.divergence = Divergence(name)

        self.current_gen = None
        self.current_dis = None
        self.log_batch_gen = []
        self.log_batch_dis = []
        self.log_epoch_gen = []
        self.log_epoch_dis = []

        self.current_real = None
        self.current_fake = None
        self.log_batch_real = []
        self.log_batch_fake = []
        self.log_epoch_real = []
        self.log_epoch_fake = []

    def log_batch_to_epoch(self):
        self.log_epoch_gen.append(np.mean(self.log_batch_gen))
        self.log_epoch_dis.append(np.mean(self.log_batch_dis))
        self.log_batch_gen = []
        self.log_batch_dis = []

        self.log_epoch_real.append(np.mean(self.log_batch_real))
        self.log_epoch_fake.append(np.mean(self.log_batch_fake))
        self.log_batch_real = []
        self.log_batch_fake = []

# utility data structure for storing performance measurements of the network
class PerformanceData:
    def __init__(self):
        self.accumulator_walk_gen = 0
        self.accumulator_walk_dis = 0
        self.epoch_walk_gen = []
        self.epoch_walk_dis = []

def run_exp(argsdict):
    # Dump the arguments so that hyperparameters are known when interpreting the data later
    with open(f"{argsdict['dataset']}_IMGS/{argsdict['divergence']}/DataHyperparameters.txt", "w") as file:
            json.dump(argsdict, file)

    # Example of usage of the code provided and recommended hyper parameters for training GANs.
    data_root = './'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_iter = argsdict['epochs'] # N training iterations
    n_critic_updates = argsdict['critic_updates'] # N critic updates per generator update
    train_batch_size = argsdict['batch_size']
    lr = argsdict['learn_rate']
    beta1 = argsdict['beta_1']
    beta2 = argsdict['beta_2']
    z_dim = argsdict['z_dimensions']
    hidden_dim = tuple(argsdict['hidden_dimensions'])

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
        for gaus in range(argsdict['number_gaussians']):
            mus.append([random.randint(0, 27) for _ in range(argsdict['Gauss_size'])])
        mus = torch.tensor(mus)
        argsdict['mus'] = mus

    # Use the GPU if you have one
    if torch.cuda.is_available():
        print("Using the GPU")
        device = torch.device("cuda")
    else:
        print("WARNING: You are about to run on cpu, and this will likely run out of memory.\nYou can try setting batch_size=1 to reduce memory usage")
        device = torch.device("cpu")

    train_loader, valid_loader, test_loader, num_samples = get_data(argsdict)
    print(f"Device: {device}")
    print(f"Number of samples: {num_samples}")
    generator = Generator(image_shape, hidden_dim[0], hidden_dim[1], z_dim, encoding).to(device)
    # generator = Generatorsvhn(z_dim, hidden_dim).to(device)
    critic = Critic(image_shape, 32, 32).to(device)
    # critic = Criticsvhn(argsdict['hidden_discri_size']).to(device)

    #TODO Adding beta seems to make total variation go to 0, why.
    #TODO In rapport talk about how finicky the whole system is
    if argsdict['optimizer']=='SGD':
        optim_critic = ss_SGD(critic.parameters(), lr=0.0001)
        optim_generator = ss_SGD(generator.parameters(), lr=0.0001)
    else:
        optim_critic = optim.Adam(critic.parameters(), lr=lr)#, betas=(beta1, beta2))
        optim_generator = optim.Adam(generator.parameters(), lr=lr)#, betas=(beta1, beta2))

    if argsdict['use_cuda']:
        Fix_Noise = Variable(torch.normal(torch.zeros(25, z_dim), torch.ones(25, z_dim))).cuda()
    else:
        Fix_Noise = Variable(torch.normal(torch.zeros(25, z_dim), torch.ones(25, z_dim)))

    # divergence used for training
    training = DivergenceData(argsdict['divergence'])

    # other divergences, only for logging if enabled
    if argsdict["divergence_all_other"]:
        other = []
        for divergence_name in ['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon']:
            if divergence_name != argsdict['divergence']:
                other.append(DivergenceData(divergence_name))

    objective=1#Initialize objective F(\theta, \omega) in the article
    # COMPLETE TRAINING PROCEDURE
    for epoch in range(n_iter):
        if argsdict['visualize']:
            real_imgs = torch.zeros([num_samples, image_shape[1], image_shape[2]])

        for i_batch, sample_batch in enumerate(train_loader):
            optim_critic.zero_grad()
            if argsdict['use_cuda']:
                real_img, label_batch = sample_batch[0].cuda(), sample_batch[1]
            else:
                real_img, label_batch = sample_batch[0], sample_batch[1]
            if argsdict['visualize']:
                real_imgs[i_batch * train_batch_size:i_batch * train_batch_size + train_batch_size] = real_img.squeeze(1)
            # Fake image
            if argsdict['use_cuda']:
                noise = Variable(torch.normal(torch.zeros(train_batch_size, z_dim), torch.ones(train_batch_size, z_dim))).cuda()
            else:
                noise = Variable(torch.normal(torch.zeros(train_batch_size, z_dim), torch.ones(train_batch_size, z_dim)))
            fake_img = generator(noise)

            # Train discriminator
            # Compute discriminator loss and real / fake statistic for training divergence
            score_dx = critic(real_img)
            score_dg = critic(fake_img)
            training.current_dis = training.divergence.D_loss(score_dx, score_dg)

            training.current_dis.backward()
            if argsdict['optimizer']=='SGD':
                optim_critic.single_step(objective=objective)
            else:
                optim_critic.step()
            training.log_batch_dis.append(training.current_dis.item())

            training.current_real, training.current_fake = training.divergence.RealFake(score_dx, score_dg)
            training.log_batch_real.append(training.current_real)
            training.log_batch_fake.append(training.current_fake)

            # Compute discriminator loss and real / fake statistic for other divergences, if enabled
            if argsdict["divergence_all_other"]:
                for item in other:
                    item.current_dis = item.divergence.D_loss(score_dx, score_dg)
                    item.log_batch_dis.append(item.current_dis.item())

                    item.current_real, item.current_fake = item.divergence.RealFake(score_dx, score_dg)
                    item.log_batch_real.append(item.current_real)
                    item.log_batch_fake.append(item.current_fake)

            # Train generator
            # Compute generator loss and real / fake statistic for training divergence
            if i_batch % n_critic_updates == 0:
                optim_generator.zero_grad()
                gen_img = generator(noise)

                if argsdict['modified_loss']:
                    score_dg = critic(gen_img)
                    # We maximize instead of minimizing
                    training.current_gen = training.divergence.G_loss_modified_sec_32(score_dg)
                else:
                    score_dg = critic(gen_img)
                    training.current_gen = training.divergence.G_loss(score_dg)

                training.current_gen.backward()
                if argsdict['optimizer']=='SGD':
                    optim_generator.single_step(objective=objective)
                else:
                    optim_generator.step()
            if argsdict['optimizer'] == 'SGD':
                objective = item.current_dis.item() #F(\theta, \omega) with the updated parameters
            training.log_batch_gen.append(training.current_gen.item())
            
            # Compute generator loss and real / fake statistic for other divergences, if enabled
            if argsdict["divergence_all_other"]:
                for item in other:
                    if argsdict['modified_loss']:
                        # We maximize instead of minimizing
                        item.current_gen = item.divergence.G_loss_modified_sec_32(score_dg)
                    else:
                        item.current_gen = item.divergence.G_loss(score_dg)
                    
                    item.log_batch_gen.append(item.current_gen.item())

        # Finished all batches within epoch, average losses over the batches and log them, then clear the batch data
        training.log_batch_to_epoch()
        if argsdict["divergence_all_other"]:
                for item in other:
                    item.log_batch_to_epoch()
        if not argsdict["divergence_all_other"]:
            print(f"Epoch {epoch:>3} of {n_iter:<3} | Generator loss: {training.log_epoch_gen[-1]:15.3f} | Discriminator loss: {training.log_epoch_dis[-1]:15.3f} | Real statistic: {training.log_epoch_real[-1]:.2f} | Fake statistic: {training.log_epoch_fake[-1]:.2f}")
        else:
            print(f"=================================================================================================")
            print(f"Epoch {epoch:>3} of {n_iter:<3}  |      Generator loss |  Discriminator loss |  Real statistic |  Fake statistic")
            print(f"-------------------------------------------------------------------------------------------------")
            print(f"{training.name:<17} | {training.log_epoch_gen[-1]:19.3f} | {training.log_epoch_dis[-1]:19.3f} | {training.log_epoch_real[-1]:15.2f} | {training.log_epoch_fake[-1]:15.2f}")
            print(f"- - - - - - - - - | - - - - - - - - - - | - - - - - - - - - - | - - - - - - - - | - - - - - - - -")
            for item in other:
                print(f"{item.name:<17} | {item.log_epoch_gen[-1]:19.3f} | {item.log_epoch_dis[-1]:19.3f} | {item.log_epoch_real[-1]:15.2f} | {item.log_epoch_fake[-1]:15.2f}")
                
        if argsdict['dataset'] == 'Gaussian':
            #A bit hacky but reset iterators
            train_loader, valid_loader, test_loader = get_data(argsdict)

        if argsdict['visualize']:
            if argsdict['use_cuda']:
                noise = Variable(torch.normal(torch.zeros(500, z_dim), torch.ones(500, z_dim))).cuda()
            else:
                noise = Variable(torch.normal(torch.zeros(500, z_dim), torch.ones(500, z_dim)))
            fake_imgs = generator(noise)
            visualize_tsne(fake_imgs, real_imgs[:500], argsdict['dataset'], argsdict['divergence'], epoch)

        with torch.no_grad():
            img = generator(Fix_Noise)

        # Saving Images
        if argsdict['modified_loss']:
            save_image(img.view(-1, image_shape[0], image_shape[1], image_shape[2]), f"{argsdict['dataset']}_IMGS/{argsdict['divergence']}/GRID_trick32%d.png" % epoch, nrow=5, normalize=True)
        else:
            save_image(img.view(-1, image_shape[0], image_shape[1], image_shape[2]), f"{argsdict['dataset']}_IMGS/{argsdict['divergence']}/GRID%d.png" % epoch, nrow=5, normalize=True)

        # Data dump
        with open(f"{argsdict['dataset']}_IMGS/{argsdict['divergence']}/DataDivergenceTraining.txt", "w") as file:
            json.dump({"divergence": training.name,
                       "gen_loss": training.log_epoch_gen,
                       "dis_loss": training.log_epoch_dis,
                       "real_stat": training.log_epoch_real,
                       "fake_stat": training.log_epoch_fake}, file)
        if argsdict["divergence_all_other"]:
            with open(f"{argsdict['dataset']}_IMGS/{argsdict['divergence']}/DataDivergenceOther.txt", "w") as file:
                info_all = []
                for item in other:
                    info_item = {"divergence": item.name,
                                "gen_loss": item.log_epoch_gen,
                                "dis_loss": item.log_epoch_dis,
                                "real_stat": item.log_epoch_real,
                                "fake_stat": item.log_epoch_fake}
                    info_all.append(info_item)
                json.dump(info_all, file)
    
        # Update the losses plot
        if epoch + 1 != n_iter:
            plot_divergence_training(argsdict['dataset'], argsdict['divergence'], show_plot=False)
            if argsdict["divergence_all_other"]:
                plot_divergence_other(argsdict['dataset'], argsdict['divergence'], show_plot=False)
            plot_real_fake_training(argsdict['dataset'], argsdict['divergence'], show_plot=False)
    
    # Epochs are over, finally display the plots
    plot_divergence_training(argsdict['dataset'], argsdict['divergence'], show_plot=True)
    if argsdict["divergence_all_other"]:
        plot_divergence_other(argsdict['dataset'], argsdict['divergence'], show_plot=True)
    plot_real_fake_training(argsdict['dataset'], argsdict['divergence'], show_plot=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Project for IFT6269 on fgans')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='Dataset you want to use. Options include MNIST, svhn, Gaussian, and CIFAR')
    parser.add_argument('--divergence', type=str, default='total_variation',
                        help='divergence to use. Options include total_variation, forward_kl, reverse_kl, pearson, hellinger, jensen_shannon, alpha_div or all')
    parser.add_argument('--divergence_all_other', action='store_false',
                        help='Logs all other divergences for comparaison')
    parser.add_argument('--Gauss_size', type=int, default='2', help='The size of the Gaussian we generate')
    parser.add_argument('--number_gaussians', type=int, default='1', help='The number of Gaussian we generate')
    parser.add_argument('--epochs', type=int, default='50', help='Number of epochs to run for training')
    parser.add_argument('--batch_size', type=int, default='64', help='Batch size for training and testing')
    parser.add_argument('--critic_updates', type=int, default='1', help='Number of critic updates per generator update')
    parser.add_argument('--learn_rate', type=float, default='1e-4', help='Learning rate')
    parser.add_argument('--beta_1', type=float, default='0.5', help='Beta 1')
    parser.add_argument('--beta_2', type=float, default='0.9', help='Beta 2')
    parser.add_argument('--z_dimensions', type=int, default='25', help='Z dimensions')
    parser.add_argument('--hidden_dimensions', nargs='+', type=int, default=[200, 200], help='Hidden dimensions')
    parser.add_argument('--modified_loss', action='store_true', help='Use the loss of section 3.2 instead of the original formulation')
    parser.add_argument('--hidden_crit_size', type=int, default=32)
    parser.add_argument('--visualize', action='store_false', help='Save visualization of the datasets using t-sne')
    parser.add_argument('--use_cuda', action='store_true', help='Use gpu')
    parser.add_argument('--optimizer', type=str, default='adams', help='The optimizer used for updating the distribution parameters. Include Adams and SGD')
    args = parser.parse_args()

    argsdict = args.__dict__
    print("=======================================================================================================================================")
    print(argsdict)
    print("=======================================================================================================================================")
    if argsdict['divergence']=='all':
        divergence=['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon','alpha_div']
        for dd in divergence:
            print(dd)
            argsdict['divergence']=dd
            run_exp(argsdict)
    else:
        run_exp(argsdict)