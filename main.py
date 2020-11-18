import os, sys
sys.path.append(os.path.abspath(os.path.join('../src/')))
from f_gan import *
import argparse


def train(argsdict):
    # Load in binarized MNIST data, separate into data loaders
    train_iter, val_iter, test_iter = get_data(argsdict)

    #size is a tuple of tot numb pixel x numb channel
    if argsdict['dataset']=="MNIST":
        img_size=784
    elif argsdict['dataset']=='CIFAR':
        img_size=(1024, 3)
    elif argsdict['dataset']=='svhn':
        img_size=(1024, 3)

    # Init model
    model = fGAN(image_size=img_size,
                 hidden_dim=400,
                 z_dim=20)

    # Init trainer
    trainer = fGANTrainer(model=model,
                          train_iter=train_iter,
                          val_iter=val_iter,
                          test_iter=test_iter,
                          viz=True)

    # Train
    trainer.train(num_epochs=25,
                  method=argsdict['divergence'],
                  G_lr=1e-4,
                  D_lr=1e-4,
                  D_steps=1)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Project for IFT6269 on fgans')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset you want to use. Options include MNIST, CelebA, and CIFAR')
    parser.add_argument('--divergence', type=str, default='forward_kl', help='divergence to use. Options include total_variation, forward_kl, reverse_kl, pearson, hellinger, jensen_shannon')
    parser.add_argument('--batch_size', type=int, default='100', help='batch size for training and testing')
    args = parser.parse_args()
    argsdict = args.__dict__
    train(argsdict)