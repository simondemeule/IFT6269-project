import torch
from torch import optim
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import get_data, find_last_run_index, anneal_function
from f_gan import Generator, Critic, Divergence
import argparse
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from plotting import plot_divergence_training, plot_divergence_other, plot_real_fake_training, plot_walk_training, plot_tsne
from single_step_SGD import *
import os.path

# Utility data structure for storing divergence data
class DivergenceData:
    def __init__(self, name, argsdict):
        self.name = name
        self.divergence = Divergence(name, argdict=argsdict)

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

# Utility data structure for storing parameter walk of the network
class WalkData:
    def __init__(self):
        self.param_init_gen = None
        self.param_init_dis = None
        self.param_final_gen = None
        self.param_final_dis = None
        
        self.walk_current_gen = None
        self.walk_current_dis = None
        self.walk_log_epoch_gen = []
        self.walk_log_epoch_dis = []

def run_exp(argsdict):
    # Give a number to this run
    run = find_last_run_index(argsdict['dataset'], argsdict['divergence']) + 1
    os.mkdir(f"experiments/{argsdict['dataset']}/{argsdict['divergence']}/{run:0>3}")

    # Dump the arguments so that hyperparameters are known when interpreting the data later
    with open(f"experiments/{argsdict['dataset']}/{argsdict['divergence']}/{run:0>3}/DataHyperparameters.txt", "w") as file:
            json.dump(argsdict, file)

    # Example of usage of the code provided and recommended hyper parameters for training GANs.
    data_root = './'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator_latent_dimensions = argsdict['generator_latent_dimensions']
    generator_hidden_dimensions = tuple(argsdict['generator_hidden_dimensions'])
    discriminator_hidden_dimensions = tuple(argsdict['discriminator_hidden_dimensions'])

    if argsdict['dataset'] in ['SVHN']:
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
            mus.append([random.randint(0, 27) for _ in range(argsdict['gaussian_size'])])
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
    generator = Generator(image_shape, generator_hidden_dimensions[0], generator_hidden_dimensions[1], generator_latent_dimensions, encoding).to(device)
    critic = Critic(image_shape, discriminator_hidden_dimensions[0], discriminator_hidden_dimensions[1]).to(device)

    #TODO Adding beta seems to make total variation go to 0, why.
    #TODO In rapport talk about how finicky the whole system is
    if argsdict['optimizer'] == 'SGD':
        optim_critic = ss_SGD(critic.parameters(), lr=0.0000001)
        optim_generator = ss_SGD(generator.parameters(), lr=0.1)
    else:
        optim_critic = optim.Adam(critic.parameters(), lr=argsdict['learn_rate'])#, betas=(argsdict['beta_1'], argsdict['beta_2']))
        optim_generator = optim.Adam(generator.parameters(), lr=argsdict['learn_rate'])#, betas=(argsdict['beta_1'], argsdict['beta_2']))

    if argsdict['use_cuda']:
        Fix_Noise = Variable(torch.normal(torch.zeros(25, generator_latent_dimensions), torch.ones(25, generator_latent_dimensions))).cuda()
    else:
        Fix_Noise = Variable(torch.normal(torch.zeros(25, generator_latent_dimensions), torch.ones(25, generator_latent_dimensions)))

    # Data for training divergence
    training = DivergenceData(argsdict['divergence'], argsdict)

    # Data for ohter divergence, if enabled
    if argsdict["divergence_all_other"]:
        other = []
        for divergence_name in ['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon']:
            if divergence_name != argsdict['divergence']:
                other.append(DivergenceData(divergence_name, argsdict))

    # Data for parameter walk, if enabled
    if argsdict['parameter_walk']:
        walk = WalkData()

    objective=1#Initialize objective F(\theta, \omega) in the article
    # COMPLETE TRAINING PROCEDURE
    for epoch in range(argsdict['epochs']):
        if argsdict['parameter_walk']:
            if epoch == 0:
                walk.param_init_gen = torch.cat(tuple(torch.flatten(x) for x in generator.parameters())).detach().clone()
                walk.param_init_dis = torch.cat(tuple(torch.flatten(x) for x in critic.parameters())).detach().clone()
            else:
                walk.param_init_gen = walk.param_final_gen
                walk.param_init_dis = walk.param_final_dis

        if argsdict['visualize']:
            real_imgs = torch.zeros([num_samples, image_shape[1], image_shape[2]])

        for i_batch, sample_batch in enumerate(train_loader):
            optim_critic.zero_grad()
            if argsdict['use_cuda']:
                real_img, label_batch = sample_batch[0].cuda(), sample_batch[1]
            else:
                real_img, label_batch = sample_batch[0], sample_batch[1]
            if argsdict['visualize']:
                real_imgs[i_batch * argsdict['batch_size']:i_batch * argsdict['batch_size'] + argsdict['batch_size']] = real_img.squeeze(1)
            # Fake image
            if argsdict['use_cuda']:
                noise = Variable(torch.normal(torch.zeros(argsdict['batch_size'], generator_latent_dimensions), torch.ones(argsdict['batch_size'], generator_latent_dimensions))).cuda()
            else:
                noise = Variable(torch.normal(torch.zeros(argsdict['batch_size'], generator_latent_dimensions), torch.ones(argsdict['batch_size'], generator_latent_dimensions)))
            fake_img = generator(noise)

            # Train discriminator
            # Compute discriminator loss and real / fake statistic for training divergence
            score_dx = critic(real_img)
            score_dg = critic(fake_img)
            training.current_dis = training.divergence.D_loss(score_dx, score_dg)

            # print(training.divergence.D_loss(score_dx, score_dg))

            training.current_dis.backward()
            if argsdict['optimizer'] == 'SGD':
                optim_critic.single_step(objective=-objective)
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
            # clipping=anneal_function('logistic', epoch, 0.5, argsdict['epochs']/4)
            # # print(clipping)
            # for p in critic.parameters():
            #     p.data.clamp_(-clipping, clipping)

            # Train generator
            # Compute generator loss and real / fake statistic for training divergence
            if i_batch % argsdict['discriminator_updates'] == 0:
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
                if argsdict['optimizer'] == 'SGD':
                    optim_generator.single_step(objective=objective)
                else:
                    optim_generator.step()

            objective = -training.current_dis.item() #F(\theta, \omega) with the updated parameters
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

        # Finished all batches within epoch

         # Calculate parameter walk if enabled
        # print(clipping)
        if argsdict['parameter_walk']:
            walk.param_final_gen = torch.cat(tuple(torch.flatten(x) for x in generator.parameters())).detach().clone()
            walk.param_final_dis = torch.cat(tuple(torch.flatten(x) for x in critic.parameters())).detach().clone()

            walk.walk_current_gen = float(np.linalg.norm(walk.param_final_gen - walk.param_init_gen))
            walk.walk_current_dis = float(np.linalg.norm(walk.param_final_dis - walk.param_init_dis))

            walk.walk_log_epoch_gen.append(walk.walk_current_gen)
            walk.walk_log_epoch_dis.append(walk.walk_current_dis)

        # Average losses over the batches and log them, then clear the batch data
        training.log_batch_to_epoch()
        if argsdict["divergence_all_other"]:
                for item in other:
                    item.log_batch_to_epoch()

        # Print iteration info
        if not argsdict["divergence_all_other"]:
            print(f"============================================================================================================================================================================================")
            if not argsdict['parameter_walk']:
                print(f"Epoch {epoch:>3} of {argsdict['epochs']:<3} | {'Generator loss:':<21}{training.log_epoch_gen[-1]:6.3f} | {'Discriminator loss:':<21}{training.log_epoch_dis[-1]:6.3f} | {'Real statistic:':<17}{training.log_epoch_real[-1]:6.2f} | {'Fake statistic:':<17}{training.log_epoch_fake[-1]:6.2f}")
            else:
                print(f"Epoch {epoch:>3} of {argsdict['epochs']:<3} | {'Generator loss:':<21}{training.log_epoch_gen[-1]:6.3f} | {'Discriminator loss:':<21}{training.log_epoch_dis[-1]:6.3f} | {'Real statistic:':<17}{training.log_epoch_real[-1]:6.2f} | {'Fake statistic:':<17}{training.log_epoch_fake[-1]:6.2f} | {'Generator walk:':<21}{walk.walk_current_gen:>6.3f} | {'Discriminator walk:':<21}{walk.walk_current_dis:>6.3f}")
        else:
            print(f"=================================================================================================")
            print(f"Epoch {epoch:>3} of {argsdict['epochs']:<3}  | {'Generator loss':>19} | {'Discriminator loss':>19} | {'Real statistic':>15} | {'Fake statistic':>15}")
            print(f"-------------------------------------------------------------------------------------------------")
            print(f"{training.name:<17} | {training.log_epoch_gen[-1]:19.3f} | {training.log_epoch_dis[-1]:19.3f} | {training.log_epoch_real[-1]:15.2f} | {training.log_epoch_fake[-1]:15.2f}")
            print(f"- - - - - - - - - | - - - - - - - - - - | - - - - - - - - - - | - - - - - - - - | - - - - - - - -")
            for item in other:
                print(f"{item.name:<17} | {item.log_epoch_gen[-1]:19.3f} | {item.log_epoch_dis[-1]:19.3f} | {item.log_epoch_real[-1]:15.2f} | {item.log_epoch_fake[-1]:15.2f}")
            if argsdict['parameter_walk']:
                print(f"-------------------------------------------------------------------------------------------------")
                print(f"                  *   {'Generator walk:':<19}{walk.walk_current_gen:>6.3f}     {'Discriminator walk:':<19}{walk.walk_current_dis:>6.3f}   *")
                
        if argsdict['dataset'] == 'Gaussian':
            #A bit hacky but reset iterators
            train_loader, valid_loader, test_loader = get_data(argsdict)

        if argsdict['visualize']:
            if argsdict['use_cuda']:
                noise = Variable(torch.normal(torch.zeros(500, generator_latent_dimensions), torch.ones(500, generator_latent_dimensions))).cuda()
            else:
                noise = Variable(torch.normal(torch.zeros(500, generator_latent_dimensions), torch.ones(500, generator_latent_dimensions)))
            fake_imgs = generator(noise)
            plot_tsne(fake_imgs, real_imgs[:500], argsdict['dataset'], argsdict['divergence'], run, epoch)

        with torch.no_grad():
            img = generator(Fix_Noise)

        # Saving Images
        if argsdict['modified_loss']:
            save_image(img.view(-1, image_shape[0], image_shape[1], image_shape[2]), f"experiments/{argsdict['dataset']}/{argsdict['divergence']}/{run:0>3}/GRID_trick32%d.png" % epoch, nrow=5, normalize=True)
        else:
            save_image(img.view(-1, image_shape[0], image_shape[1], image_shape[2]), f"experiments/{argsdict['dataset']}/{argsdict['divergence']}/{run:0>3}/GRID%d.png" % epoch, nrow=5, normalize=True)

        # Data dump
        with open(f"experiments/{argsdict['dataset']}/{argsdict['divergence']}/{run:0>3}/DataDivergenceTraining.txt", "w") as file:
            json.dump({"divergence": training.name,
                       "gen_loss": training.log_epoch_gen,
                       "dis_loss": training.log_epoch_dis,
                       "real_stat": training.log_epoch_real,
                       "fake_stat": training.log_epoch_fake}, file)
        if argsdict["divergence_all_other"]:
            with open(f"experiments/{argsdict['dataset']}/{argsdict['divergence']}/{run:0>3}/DataDivergenceOther.txt", "w") as file:
                info_all = []
                for item in other:
                    info_item = {"divergence": item.name,
                                "gen_loss": item.log_epoch_gen,
                                "dis_loss": item.log_epoch_dis,
                                "real_stat": item.log_epoch_real,
                                "fake_stat": item.log_epoch_fake}
                    info_all.append(info_item)
                json.dump(info_all, file)
        if argsdict["parameter_walk"]:
            with open(f"experiments/{argsdict['dataset']}/{argsdict['divergence']}/{run:0>3}/DataParameterWalk.txt", "w") as file:
                info = {"gen_walk": walk.walk_log_epoch_gen, "dis_walk": walk.walk_log_epoch_dis}
                json.dump(info, file)

        # Update the losses plot
        if epoch + 1 != argsdict['epochs'] and argsdict['plot']:
            plot_divergence_training(argsdict['dataset'], argsdict['divergence'], run, show_plot=False)
            if argsdict["divergence_all_other"]:
                plot_divergence_other(argsdict['dataset'], argsdict['divergence'], run, show_plot=False)
            plot_real_fake_training(argsdict['dataset'], argsdict['divergence'], run, show_plot=False)
            if argsdict["parameter_walk"]:
                plot_walk_training(argsdict['dataset'], argsdict['divergence'], run, show_plot=False)
    
    # Epochs are over, finally display the plots
    plot_divergence_training(argsdict['dataset'], argsdict['divergence'], run, show_plot=True)
    if argsdict["divergence_all_other"]:
        plot_divergence_other(argsdict['dataset'], argsdict['divergence'], run, show_plot=True)
    plot_real_fake_training(argsdict['dataset'], argsdict['divergence'], run, show_plot=True)
    if argsdict["parameter_walk"]:
        plot_walk_training(argsdict['dataset'], argsdict['divergence'], run, show_plot=True)

    # plot_divergence_training(argsdict['dataset'], argsdict['divergence'], run, show_plot=show_plot)
    # if argsdict["divergence_all_other"]:
    #     plot_divergence_other(argsdict['dataset'], argsdict['divergence'], run, show_plot=show_plot)
    # plot_real_fake_training(argsdict['dataset'], argsdict['divergence'], run, show_plot=show_plot)
    # if argsdict["parameter_walk"]:
    #     plot_walk_training(argsdict['dataset'], argsdict['divergence'], run, show_plot=show_plot)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Project for IFT6269 on fgans')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='Dataset you want to use. Options include MNIST, SVHN, Gaussian, and CIFAR')
    parser.add_argument('--divergence', type=str, default='total_variation',
                        help='divergence to use. Options include total_variation, forward_kl, reverse_kl, pearson, hellinger, jensen_shannon, alpha_div, piecewise or all')
    parser.add_argument('--divergence_all_other', action='store_true',
                        help='Logs all other divergences for comparaison')
    parser.add_argument('--parameter_walk', action='store_false', help='Log the L2 norm of the parameter change at each epoch while training')
    parser.add_argument('--gaussian_size', type=int, default='2', help='The size of the Gaussian we generate')
    parser.add_argument('--gaussian_number', type=int, default='1', help='The number of Gaussian we generate')
    parser.add_argument('--epochs', type=int, default='50', help='Number of epochs to run for training')
    parser.add_argument('--batch_size', type=int, default='64', help='Batch size for training and testing')
    parser.add_argument('--learn_rate', type=float, default='1e-4', help='Learning rate')
    parser.add_argument('--beta_1', type=float, default='0.5', help='Beta 1')
    parser.add_argument('--beta_2', type=float, default='0.9', help='Beta 2')

    parser.add_argument('--use_cnn_generator', action='store_true', help='whether to use the cnn generator or the linear one')
    parser.add_argument('--generator_latent_dimensions', type=int, default='25', help='Latent dimensions of generator (Int)')
    parser.add_argument('--generator_hidden_dimensions', nargs='+', type=int, default=[64, 64], help='Hidden dimensions of generator (Int array)')
    parser.add_argument('--discriminator_hidden_dimensions', nargs='+', type=int, default=[32, 32], help='Hidden dimensions of discriminator (Int array)')
    parser.add_argument('--discriminator_updates', type=int, default='1', help='Number of critic updates per generator update')

    parser.add_argument('--modified_loss', action='store_true', help='Use the loss of section 3.2 instead of the original formulation')
    parser.add_argument('--visualize', action='store_false', help='Save visualization of the datasets using t-sne')
    parser.add_argument('--use_cuda', action='store_true', help='Use gpu')
    parser.add_argument('--plot', action='store_true', help='create the plots')
    parser.add_argument('--optimizer', type=str, default='adams', help='The optimizer used for updating the distribution parameters. Include Adams and SGD')

    parser.add_argument('--falseDiv', type=str, default='hellinger', help='for piecewise divergence divergence to use when Tx is lower then threshold')
    parser.add_argument('--trueDiv', type=str, default='forward_kl', help='for piecewise divergence divergence to use when Tx is higher then threshold')
    args = parser.parse_args()

    argsdict = args.__dict__
    print("=================================================================================================")
    print(argsdict)
    print("=================================================================================================")
    #Total Var: 128 128 64 64
    #Forward 3 4K 4K 2K 2K
    #Reverse 3 4K 4K 2K 2K
    #Pearson 3 4K 4K 2K 2K
    #hellinger 3 4K 4K 2K 2K
    #jesnsen shannon KL 128 128 32 32
    #piecewise  3 4K 4K 2K 2K

    if argsdict['divergence'] == 'all':
        divergences = ['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon','piecewise']
        for divergence in divergences:
            print(divergence)
            argsdict['divergence'] = divergence
            run_exp(argsdict)
    else:
        run_exp(argsdict)