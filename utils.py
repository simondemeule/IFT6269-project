import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random    
import os.path

def find_last_run_index(dataset, divergence):
    """ Finds the index of the last finished run, for a specific dataset and divergence """
    run = 0
    while os.path.isdir(f"experiments/{dataset}/{divergence}/{run:0>3}"):
        run += 1
    # returns -1 if no files are found starting at zero
    return run - 1

def to_var(x):
    """ Make a tensor cuda-erized and requires gradient """
    return to_cuda(x).requires_grad_()

def to_cuda(x):
    """ Cuda-erize a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def get_data(argsdict):
    """ Load data for binared MNIST """
    torch.manual_seed(3435)

    BATCH_SIZE = argsdict['batch_size']

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    transform = transforms.Compose((
        transforms.ToTensor(),
        normalize))

    if argsdict['dataset']=="MNIST":
        # Download our data
        train_dataset = datasets.MNIST(root='data/',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)

        test_dataset = datasets.MNIST(root='data/',
                                       train=False,
                                       transform=transforms.ToTensor())
        train_img = torch.stack([torch.bernoulli(d[0]) for d in train_dataset])
        train_label = torch.LongTensor([d[1] for d in train_dataset])

        test_img = torch.stack([torch.bernoulli(d[0]) for d in test_dataset])
        test_label = torch.LongTensor([d[1] for d in test_dataset])

        # MNIST has no official train dataset so use last 10000 as validation
        val_img = train_img[-10000:].clone()
        val_label = train_label[-10000:].clone()

        train_img = train_img[:-10000]
        train_label = train_label[:-10000]

        # Create data loaders
        train = torch.utils.data.TensorDataset(train_img, train_label)
        val = torch.utils.data.TensorDataset(val_img, val_label)
        test = torch.utils.data.TensorDataset(test_img, test_label)

    elif argsdict['dataset']=="CelebA":
        train_dataset = datasets.CelebA(root='data/',
                                       split='train',
                                       transform=transforms.ToTensor(),
                                       download=True)

        test_dataset = datasets.CelebA(root='data/',
                                      split='valid',
                                      transform=transforms.ToTensor())
                                      
    elif argsdict['dataset']=="CIFAR":
        train = datasets.CIFAR10(root='data/',
                                       train=True,
                                       transform=transforms.ToTensor(),
                                       download=True)

        val = datasets.CIFAR10(root='data/',
                                      train=False,
                                      transform=transforms.ToTensor())

        test = datasets.CIFAR10(root='data/',
                               train=False,
                               transform=transforms.ToTensor())
    elif argsdict['dataset']=="SVHN":
        train = datasets.SVHN('data/', split='train', download=True, transform=transform)
        val = datasets.SVHN('data/', split='train', download=True, transform=transform)
        test = datasets.SVHN('data/', split='test', download=True, transform=transform)

    elif argsdict['dataset']=='Gaussian':
        train_iter=GaussianGen(argsdict, BATCH_SIZE, 1000)
        val_iter=GaussianGen(argsdict, BATCH_SIZE, 1000)
        test_iter=GaussianGen(argsdict, BATCH_SIZE, 1000)
        return train_iter, val_iter, test_iter

    train_iter = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

    return train_iter, val_iter, test_iter, len(train)

def GaussianGen(argsdict, batch_size, Total):
    #Todo possibly a more efficient way to do this
    #TODO Add argument for number of point generated each time
    for i in range(int(Total/batch_size)):
        bb=torch.zeros((batch_size, 1, 28, 28))
        #Choose random gaussian
        for j in range(batch_size):
            grid=torch.zeros(1, 28, 28)
            for k in range(200):
                gaus=random.randint(0, argsdict['number_gaussians']-1)
                mu=argsdict['mus'][gaus]

                point=torch.round(torch.randn((argsdict['Gauss_size']))+mu)
                # print(point)
                point=torch.clip(point, 0, 27)
                grid[0, int(point[0]), int(point[1])]=1
            bb[j]=grid
        # print(bb[0])
        yield bb, torch.ones(2)