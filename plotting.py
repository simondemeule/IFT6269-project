# For simple plots
import matplotlib.pyplot as plt
import json
import os.path

# TSNE related
import torch
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns

# Mural related
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from PIL import Image
import torch
import math

divergence_names = ['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon']

divergence_colors = {   'total_variation':  'blue',
                        'forward_kl':       'orange',
                        'reverse_kl':       'purple',
                        'pearson':          'green',
                        'hellinger':        'pink',
                        'jensen_shannon':   'gray',
                        'alpha_div':        'red'}

image_shapes = {'SVHN':         (3, 32, 32),
                'CIFAR':        (3, 32,32),
                'MNIST':        (1, 28, 28),
                'Gaussian':     (1, 28, 28)}

# Converts the older Losses.txt format to the newer DataDivergenceTraining.txt format
def convert_from_legacy(dataset, run, divergence):
    try:
        with open(f"experiments/{dataset}/{divergence}/{run:0>3}/Losses.txt", "r") as file_legacy:
            try:
                with open(f"experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt", "w") as file_converted:
                    data_legacy = json.load(file_legacy)
                    data_converted = {}
                    data_converted['divergence'] = divergence
                    data_converted['gen_loss'] = data_legacy['Gen_Loss']
                    data_converted['dis_loss'] = data_legacy['Discri_Loss']
                    if 'real_stat' in data_legacy:
                        data_converted['real_stat'] = data_legacy['real_stat']
                    if 'fake_stat' in data_legacy:
                        data_converted['fake_stat'] = data_legacy['fake_stat']
                    json.dump(data_converted, file_converted)
                print(f"Conversion succeeded: created file experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt")
            except:
                print(f"Conversion failed: error during parsing or unable to write to experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt")
    except:
        print(f"Conversion failed: unable to open experiments/{dataset}/{divergence}/{run:0>3}/Losses.txt")

# Plots the evolution of the training divergence
# This data is generated from a single run, trained on a single divergence
def plot_divergence_training(dataset, divergence, run, show_plot=True):
    try:
        with open(f"experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt", "r") as file:
            training = json.load(file)
            file.close
    except:
        print(f"Unable to open experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt. If you have a legacy Losses.txt log, you can attempt to convert it with convert_from_legacy.")
        return

    epochs = [i for i in range(len(training['gen_loss']))]
    
    if not show_plot:
        plt.ioff()

    fig = plt.figure()

    plt.plot(epochs, training['gen_loss'], label='Generator Divergence', color=divergence_colors[divergence])
    plt.plot(epochs, training['dis_loss'], ':', label='Discriminator Divergence', color=divergence_colors[divergence])
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Evolution of Training Generator and Discriminator Losses\n({dataset} Trained With {divergence})')

    plt.savefig(f"experiments/{dataset}/{divergence}/{run:0>3}/PlotDivergenceTraining.svg")

    if not show_plot:
        plt.close(fig)
    else:
        plt.show()

# Plots the evolution of the other divergences, given the model is trained with a specific divergence
# This data is generated from a single run, trained on a single divergence
def plot_divergence_other(dataset, divergence, run, show_plot=True):
    try:
        with open(f"experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt", "r") as file:
            training = json.load(file)
            file.close
    except:
        print(f"Unable to open experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt\nFile may be displaced or deleted, or you may be using the older Losses.txt format.\nIn this case, you must run the training session again to generate the necessary data, as the legacy format does not contain it.")
        return
    
    try:
        with open(f"experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceOther.txt", "r") as file:
            other = json.load(file)
            file.close
    except:
        print(f"Unable to open experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceOther.txt\nFile may be displaced or deleted, or you may be using the older Losses.txt format.\nIn this case, you must run the training session again to generate the necessary data, as the legacy format does not contain it.")
        return
    
    if show_plot == 0:
        plt.ioff()

    # Generator
    fig = plt.figure()

    epochs = [i for i in range(len(training['gen_loss']))]
    plt.plot(epochs, training['gen_loss'], label=divergence, color=divergence_colors[divergence])
    for item in other:
        epochs = [i for i in range(len(item['gen_loss']))]
        plt.plot(epochs, item['gen_loss'], ':', label=item['divergence'], color=divergence_colors[item['divergence']])
    
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Evolution of Other Generator Losses\n({dataset} Trained With {divergence})')

    plt.savefig(f"experiments/{dataset}/{divergence}/{run:0>3}/PlotDivergenceOtherGenerator.svg")

    if show_plot == 0:
        plt.close(fig)
    else:
        plt.show()

    # Discriminator
    fig = plt.figure()

    epochs = [i for i in range(len(training['dis_loss']))]
    plt.plot(epochs, training['dis_loss'], label=divergence, color=divergence_colors[divergence])
    for item in other:
        epochs = [i for i in range(len(item['dis_loss']))]
        plt.plot(epochs, item['dis_loss'], ':', label=item['divergence'], color=divergence_colors[item['divergence']])
    
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Evolution of Other Discriminator Losses\n({dataset} Trained With {divergence})')

    plt.savefig(f"experiments/{dataset}/{divergence}/{run:0>3}/PlotDivergenceOtherDiscriminator.svg")

    if show_plot == 0:
        plt.close(fig)
    else:
        plt.show()

# Plots the evolution of all divergences when used individually for training
# This data is generated from multiple runs, each trained with its own divergence
def plot_divergence_all(dataset, divergences, runs, show_plot=True):
    # Generators
    if not show_plot:
        plt.ioff()

    fig = plt.figure()

    for (divergence, run) in zip(divergences, runs):
        try:
            with open(f"experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt", "r") as file:
                training = json.load(file)
                file.close
        except:
            print(f"Unable to open experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt; skipping entry.\nFile may be displaced or deleted")
            continue

        epochs = [i for i in range(len(training['gen_loss']))]
        plt.plot(epochs, training['gen_loss'], label=divergence, color=divergence_colors[divergence])

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Evolution of All Generator Losses\n({dataset} Trained With Respective Divergences)')

    plt.savefig(f"experiments/{dataset}/PlotDivergenceAllGenerator.svg")

    if not show_plot:
        plt.close(fig)
    else:
        plt.show()

    # Discriminators
    if not show_plot:
        plt.ioff()

    fig = plt.figure()
    
    for (divergence, run) in zip(divergences, runs):
        try:
            with open(f"experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt", "r") as file:
                training = json.load(file)
                file.close
        except:
            print(f"Unable to open experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt; skipping entry.\nFile may be displaced or deleted")
            continue

        epochs = [i for i in range(len(training['gen_loss']))]
        plt.plot(epochs, training['gen_loss'], label=divergence, color=divergence_colors[divergence])

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Evolution of All Discriminator Losses\n({dataset} Trained With Respective Divergences)')

    plt.savefig(f"experiments/{dataset}/PlotDivergenceAllDiscriminator.svg")

    if not show_plot:
        plt.close(fig)
    else:
        plt.show()


# Plots the evolution of real / fake statistics for a given training divergence
# This data is generated from a single run, trained on a single divergence
def plot_real_fake_training(dataset, divergence, run, show_plot=True):
    try:
        with open(f"experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt", "r") as file:
            training = json.load(file)
            file.close
    except:
        print(f"Unable to open experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt. If you have a legacy Losses.txt log, you can attempt to convert it with convert_from_legacy.")
        return

    if (not 'real_stat' in training) or (not 'fake_stat' in training):
        print(f"Data experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt contains no real / fake statistics.\nIf you converted this data from a legacy format, this likely means the legacy format does not contain the real / fake statistics: you will need to run a training session again.")
        return

    epochs = [i for i in range(len(training['real_stat']))]
    
    if not show_plot:
        plt.ioff()

    fig = plt.figure()

    plt.plot(epochs, training['real_stat'], label='Real Statistic', color=divergence_colors[divergence])
    plt.plot(epochs, training['fake_stat'], ':', label='Fake Statistic', color=divergence_colors[divergence])
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Evolution of Real / Fake Statistics\n({dataset} Trained With {divergence})')

    plt.savefig(f"experiments/{dataset}/{divergence}/{run:0>3}/PlotStatTrainingRealFake.svg")

    if not show_plot:
        plt.close(fig)
    else:
        plt.show()

# Plots the evolution of real / fake statistics for all training divergences
# This data is generated from multiple runs, each trained with its own divergence
def plot_real_fake_all(dataset, divergences, runs, show_plot=True):
    # Real statistic
    if not show_plot:
        plt.ioff()

    fig = plt.figure()
    
    for (divergence, run) in zip(divergences, runs):
        try:
            with open(f"experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt", "r") as file:
                training = json.load(file)
                file.close
        except:
            print(f"Unable to open experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt; skipping entry.\nFile may be displaced or deleted.")
            continue

        if (not 'real_stat' in training) or (not 'fake_stat' in training):
            print(f"Data experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt contains no real / fake statistics; skipping entry.\nIf you converted this data from a legacy format, this likely means the legacy format does not contain the real / fake statistics: you will need to run a training session again.")
            continue

        epochs = [i for i in range(len(training['real_stat']))]
        plt.plot(epochs, training['real_stat'], label=divergence, color=divergence_colors[divergence])

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Evolution of Real Statistics\n({dataset} Trained With Respective Divergences)')

    plt.savefig(f"experiments/{dataset}/PlotStatAllReal.svg")

    if not show_plot:
        plt.close(fig)
    else:
        plt.show()

    # Fake statistic
    if not show_plot:
        plt.ioff()

    fig = plt.figure()

    for (divergence, run) in zip(divergences, runs):
        try:
            with open(f"experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt", "r") as file:
                training = json.load(file)
                file.close
        except:
            print(f"Unable to open experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt; skipping entry.\nFile may be displaced or deleted.")
            continue

        if (not 'real_stat' in training) or (not 'fake_stat' in training):
            print(f"Data experiments/{dataset}/{divergence}/{run:0>3}/DataDivergenceTraining.txt contains no real / fake statistics; skipping entry.\nIf you converted this data from a legacy format, this likely means the legacy format does not contain the real / fake statistics: you will need to run a training session again.")
            continue

        epochs = [i for i in range(len(training['fake_stat']))]
        plt.plot(epochs, training['fake_stat'], label=divergence, color=divergence_colors[divergence])

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Evolution of Fake Statistics\n({dataset} Trained With Respective Divergences)')

    plt.savefig(f"experiments/{dataset}/PlotStatAllFake.svg")

    if not show_plot:
        plt.close(fig)
    else:
        plt.show()

# Plots the evolution of the parameter step length
# This is for a single training divergence
def plot_walk_training(dataset, divergence, run, show_plot=True):
    try:
        with open(f"experiments/{dataset}/{divergence}/{run:0>3}/DataParameterWalk.txt", "r") as file:
            training = json.load(file)
            file.close
    except:
        print(f"Unable to open experiments/{dataset}/{divergence}/{run:0>3}/DataParameterWalk.txt.")
        return

    epochs = [i for i in range(len(training['gen_walk']))]
    
    if not show_plot:
        plt.ioff()

    fig = plt.figure()

    plt.plot(epochs, training['gen_walk'], label='Generator Walk', color=divergence_colors[divergence])
    plt.plot(epochs, training['dis_walk'], ':', label='Discriminator Walk', color=divergence_colors[divergence])
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Step L2 Norm')
    plt.title(f'Evolution of Step Length\n({dataset} Trained With {divergence})')

    plt.savefig(f"experiments/{dataset}/{divergence}/{run:0>3}/PlotStatTrainingWalk.svg")

    if not show_plot:
        plt.close(fig)
    else:
        plt.show()

# Plots the evolution of the parameter step length
# This is for a set of divergences
def plot_walk_all(dataset, divergences, runs, show_plot=True):
    # Generator
    if not show_plot:
        plt.ioff()

    fig = plt.figure()
    
    for (divergence, run) in zip(divergences, runs):
        try:
            with open(f"experiments/{dataset}/{divergence}/{run:0>3}/DataParameterWalk.txt", "r") as file:
                training = json.load(file)
                file.close
        except:
            print(f"Unable to open experiments/{dataset}/{divergence}/{run:0>3}/DataParameterWalk.txt; skipping entry.\nFile may be displaced or deleted.")
            continue

        epochs = [i for i in range(len(training['gen_walk']))]
        plt.plot(epochs, training['gen_walk'], label=divergence, color=divergence_colors[divergence])

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Step L2 Norm')
    plt.title(f'Evolution of Generator Step Length\n({dataset} Trained With Respective Divergences)')

    plt.savefig(f"experiments/{dataset}/PlotStatAllWalkGenerator.svg")

    if not show_plot:
        plt.close(fig)
    else:
        plt.show()

    # Discriminator
    if not show_plot:
        plt.ioff()

    fig = plt.figure()
    
    for (divergence, run) in zip(divergences, runs):
        try:
            with open(f"experiments/{dataset}/{divergence}/{run:0>3}/DataParameterWalk.txt", "r") as file:
                training = json.load(file)
                file.close
        except:
            print(f"Unable to open experiments/{dataset}/{divergence}/{run:0>3}/DataParameterWalk.txt; skipping entry.\nFile may be displaced or deleted.")
            continue

        epochs = [i for i in range(len(training['dis_walk']))]
        plt.plot(epochs, training['dis_walk'], label=divergence, color=divergence_colors[divergence])

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Step L2 Norm')
    plt.title(f'Evolution of Discriminator Step Length\n({dataset} Trained With Respective Divergences)')

    plt.savefig(f"experiments/{dataset}/PlotStatAllWalkDiscriminator.svg")

    if not show_plot:
        plt.close(fig)
    else:
        plt.show()

# Plots a TSNE representation
def plot_tsne(fake_img, real_img, dataset, divergence, run, epoch):
    # Reshaping images and concatenating
    imgs = torch.cat([fake_img.reshape(fake_img.shape[0], -1).cpu().detach(), real_img.reshape(real_img.shape[0], -1).cpu().detach()])
    y = ['Generated' for _ in range(fake_img.shape[0])] + ['Real' for _ in range(real_img.shape[0])]
    # TSNE
    tsne_obj = TSNE(n_components=2).fit_transform(imgs)
    tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                            'Y': tsne_obj[:, 1],
                            'Images': y})
    sns_plot = sns.scatterplot(x="X", y="Y",
                               hue="Images",
                               palette=['purple', 'red'],
                               legend='auto',
                               data=tsne_df)
    sns_plot.figure.savefig(f"experiments/{dataset}/{divergence}/{run:0>3}/TSNEVIZ%d.png" % epoch)
    # Closing because I hate matplotlib its a piece of garbage
    plt.close(sns_plot.figure)

# Plots mural from decoded latent space samples
def plot_mural(dataset, divergences, runs, epoch_total, epoch_increment):
    mural = None
    epoch_views = math.floor(epoch_total / epoch_increment)
    mural = None
    for (divergence, run) in zip(divergences, runs):
        if not os.path.exists(f'experiments/{dataset}/{divergence}/{run:0>3}/GRID0.svg'):
            print(f"Grid for epoch 0 is missing; skipping. experiments/{dataset}/{divergence}/{run:0>3}/GRID0.svg does not exist.")
            continue
        for epoch in range(0, epoch_total, epoch_increment):
            if os.path.exists(f'experiments/{dataset}/{divergence}/{run:0>3}/GRID{epoch}.svg'):
                try:
                    grid = Image.open(f'experiments/{dataset}/{divergence}/{run:0>3}/GRID{epoch}.svg')
                    grid = TF.to_tensor(grid).unsqueeze(0)
                    if mural is None:
                        mural = grid
                        image_shape = grid.shape
                    else:
                        mural = torch.cat([mural, grid], axis=0)
                except:
                    print(f"Error while stitching mural; stopped at experiments/{dataset}/{divergence}/{run:0>3}/GRID{epoch}.svg.")
                    return
            else:
                print(f"Grid for epoch {epoch} is missing; aborting. experiments/{dataset}/{divergence}/{run:0>3}/GRID{epoch}.svg does not exist.")
                return

    save_image(mural.view(-1, 3, image_shape[2], image_shape[3]), f"experiments/{dataset}/PlotMural.png", nrow=epoch_views)