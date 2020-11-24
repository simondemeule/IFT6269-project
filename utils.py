import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    BATCH_SIZE=argsdict['batch_size']

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

    elif argsdict['dataset']=="svhn":
        train = datasets.SVHN('data/', split='train', download=True, transform=transform)
        val = datasets.SVHN('data/', split='train', download=True, transform=transform)
        test = datasets.SVHN('data/', split='test', download=True, transform=transform)


    #TODO CHANGE HERE
    # Use greyscale values as sampling probabilities to get back to [0,1]




    train_iter = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

    return train_iter, val_iter, test_iter

def visualize_tsne(fake_img, real_img, argsdict, epoch):
    """Visualizing tsn"""
    #Reshaping images and concatenating
    imgs = torch.cat([fake_img.reshape(fake_img.shape[0], -1).cpu().detach(), real_img.reshape(real_img.shape[0], -1).cpu().detach()])
    y = ['Generated' for _ in range(fake_img.shape[0])] + ['Real' for _ in range(real_img.shape[0])]
    #tsne
    tsne_obj = TSNE(n_components=2).fit_transform(imgs)
    tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                            'Y': tsne_obj[:, 1],
                            'Images': y})
    sns_plot = sns.scatterplot(x="X", y="Y",
                               hue="Images",
                               palette=['purple', 'red'],
                               legend='auto',
                               data=tsne_df)
    sns_plot.figure.savefig(f"{argsdict['dataset']}_IMGS/{argsdict['divergence']}/TSNEVIZ%d.png" % epoch)
    #Closing because I hate matplotlib its a piece of garbage
    plt.close(sns_plot.figure)