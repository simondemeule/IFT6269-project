# For simple plots
import matplotlib.pyplot as plt
import json

# TSNE related
import torch
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns

divergence_colors = {   'total_variation': 'blue',
                        'forward_kl': 'orange',
                        'reverse_kl': 'purple',
                        'pearson': 'green',
                        'hellinger': 'pink',
                        'jensen_shannon': 'gray',
                        'alpha_div': 'red'}

# Plots the evolution of the training divergence
# This data is generated from a single run, trained on a single divergence
def plot_divergence_training(dataset, divergence, show_plot=True):
    with open(f"{dataset}_IMGS/{divergence}/DataDivergenceTraining.txt", "r") as file:
        training = json.load(file)
        file.close

    epochs = [i for i in range(len(training['gen_loss']))]
    
    if not show_plot:
        plt.ioff()

    fig = plt.figure()

    plt.plot(epochs, training['gen_loss'], label='Generator Divergence', color=divergence_colors[divergence])
    plt.plot(epochs, training['dis_loss'], ':', label='Discriminator Divergence', color=divergence_colors[divergence])
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Divergence')
    plt.title(f'Evolution of the Generator and Discriminator Training Divergence\n({dataset} Trained With {divergence})')

    plt.savefig(f"{dataset}_IMGS/{divergence}/PlotDivergenceTraining.png")

    if not show_plot:
        plt.close(fig)
    else:
        plt.show()

# Plots the evolution of the other divergences, given the model is trained with a specific divergence
# This data is generated from a single run, trained on a single divergence
def plot_divergence_other(dataset, divergence, show_plot=True):
    with open(f"{dataset}_IMGS/{divergence}/DataDivergenceTraining.txt", "r") as file:
        training = json.load(file)
        file.close

    with open(f"{dataset}_IMGS/{divergence}/DataDivergenceOther.txt", "r") as file:
        other = json.load(file)
        file.close

    epochs = [i for i in range(len(training['gen_loss']))]
    
    if show_plot == 0:
        plt.ioff()

    # Generator
    fig = plt.figure()

    plt.plot(epochs, training['gen_loss'], label=divergence, color=divergence_colors[divergence])
    for item in other:
        plt.plot(epochs, item['gen_loss'], ':', label=item['divergence'], color=divergence_colors[item['divergence']])
    
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Divergence')
    plt.title(f'Evolution of Other Generator Divergences\n({dataset} Trained With {divergence})')

    plt.savefig(f"{dataset}_IMGS/{divergence}/PlotDivergenceOtherGenerator.png")

    if show_plot == 0:
        plt.close(fig)
    else:
        plt.show()

    # Discriminator
    fig = plt.figure()

    plt.plot(epochs, training['dis_loss'], label=divergence, color=divergence_colors[divergence])
    for item in other:
        plt.plot(epochs, item['dis_loss'], ':', label=item['divergence'], color=divergence_colors[item['divergence']])
    
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Divergence')
    plt.title(f'Evolution of Other Discriminator Divergences\n({dataset} Trained With {divergence})')

    plt.savefig(f"{dataset}_IMGS/{divergence}/PlotDivergenceOtherDiscriminator.png")

    if show_plot == 0:
        plt.close(fig)
    else:
        plt.show()

# Plots the evolution of real / fake statistics
# This data is generated from a single run, trained on a single divergence
def plot_real_fake_training(dataset, divergence, show_plot=True):
    with open(f"{dataset}_IMGS/{divergence}/DataDivergenceTraining.txt", "r") as file:
        training = json.load(file)
        file.close

    epochs = [i for i in range(len(training['gen_loss']))]
    
    if not show_plot:
        plt.ioff()

    fig = plt.figure()

    plt.plot(epochs, training['real_stat'], label='Real Statistic', color=divergence_colors[divergence])
    plt.plot(epochs, training['fake_stat'], ':', label='Fake Statistic', color=divergence_colors[divergence])
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Evolution of the Real / Fake Statistics\n({dataset} Trained With {divergence})')

    plt.savefig(f"{dataset}_IMGS/{divergence}/PlotRealFake.png")

    if not show_plot:
        plt.close(fig)
    else:
        plt.show()

# Plots the evolution of all divergences when used individually for training
# This data is generated from multiple runs, each trained with its own divergence
def plot_divergence_all(dataset, show_plot=True):
    pass

# Plots a TSNE representation
def visualize_tsne(fake_img, real_img, dataset, divergence, epoch):
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
    sns_plot.figure.savefig(f"{dataset}_IMGS/{divergence}/TSNEVIZ%d.png" % epoch)
    # Closing because I hate matplotlib its a piece of garbage
    plt.close(sns_plot.figure)