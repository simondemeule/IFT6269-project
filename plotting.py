# For simple plots
import matplotlib.pyplot as plt
import json
import os.path

# TSNE related
import torch
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns

divergence_names = ['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon']
divergence_colors = {   'total_variation': 'blue',
                        'forward_kl': 'orange',
                        'reverse_kl': 'purple',
                        'pearson': 'green',
                        'hellinger': 'pink',
                        'jensen_shannon': 'gray',
                        'alpha_div': 'red'}

# Converts the older Losses.txt format to the newer DataDivergenceTraining.txt format
# If no divergence is specified, this will attempt to convert all divergences for the given dataset for which there is no new format
def convert_from_legacy(dataset, divergence=None):
    if divergence is not None:
        with open(f"{dataset}_IMGS/{divergence}/Losses.txt", "r") as file_legacy:
            with open(f"{dataset}_IMGS/{divergence}/DataDivergenceTraining.txt", "w") as file_converted:
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
    else:
        for divergence_iter in divergence_names:
            if os.path.isfile(f"{dataset}_IMGS/{divergence_iter}/Losses.txt") and (not os.path.isfile(f"{dataset}_IMGS/{divergence_iter}/DataDivergenceTraining.txt")):
                print(f"Found legacy conversion candidate: {dataset}_IMGS/{divergence_iter}/Losses.txt. Attempting to convert...")
                try:
                    convert_from_legacy(dataset, divergence=divergence_iter)
                    print(f"Conversion succeeded: created file {dataset}_IMGS/{divergence_iter}/DataDivergenceTraining.txt")
                except Exception as e:
                    print(f"Conversion failed. {e}")

# Plots the evolution of the training divergence
# This data is generated from a single run, trained on a single divergence
def plot_divergence_training(dataset, divergence, show_plot=True):
    try:
        with open(f"{dataset}_IMGS/{divergence}/DataDivergenceTraining.txt", "r") as file:
            training = json.load(file)
            file.close
    except:
        print(f"Unable to open {dataset}_IMGS/{divergence}/DataDivergenceTraining.txt.\nSearching for a legacy format Losses.txt to convert...")
        try:
            convert_from_legacy(dataset, divergence=divergence)
        except:
            print("Conversion failed.")
            return
        print(f"Conversion succeeded: created file {dataset}_IMGS/{divergence}/DataDivergenceTraining.txt. Retrying...")
        # Recursive retry
        return plot_divergence_training(dataset, divergence, show_plot=show_plot)

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

    plt.savefig(f"{dataset}_IMGS/{divergence}/PlotDivergenceTraining.png")

    if not show_plot:
        plt.close(fig)
    else:
        plt.show()

# Plots the evolution of the other divergences, given the model is trained with a specific divergence
# This data is generated from a single run, trained on a single divergence
def plot_divergence_other(dataset, divergence, show_plot=True):
    try:
        with open(f"{dataset}_IMGS/{divergence}/DataDivergenceTraining.txt", "r") as file:
            training = json.load(file)
            file.close
    except:
        print(f"Unable to open {dataset}_IMGS/{divergence}/DataDivergenceTraining.txt\nFile may be displaced or deleted, or you may be using the older Losses.txt format.\nIn this case, you must run the training session again to generate the necessary data, as the legacy format does not contain it.")
        return
    
    try:
        with open(f"{dataset}_IMGS/{divergence}/DataDivergenceOther.txt", "r") as file:
            other = json.load(file)
            file.close
    except:
        print(f"Unable to open {dataset}_IMGS/{divergence}/DataDivergenceOther.txt\nFile may be displaced or deleted, or you may be using the older Losses.txt format.\nIn this case, you must run the training session again to generate the necessary data, as the legacy format does not contain it.")
        return

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
    plt.ylabel('Loss')
    plt.title(f'Evolution of Other Generator Losses\n({dataset} Trained With {divergence})')

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
    plt.ylabel('Loss')
    plt.title(f'Evolution of Other Discriminator Losses\n({dataset} Trained With {divergence})')

    plt.savefig(f"{dataset}_IMGS/{divergence}/PlotDivergenceOtherDiscriminator.png")

    if show_plot == 0:
        plt.close(fig)
    else:
        plt.show()

# Plots the evolution of all divergences when used individually for training
# This data is generated from multiple runs, each trained with its own divergence
def plot_divergence_all(dataset, show_plot=True):
    print("Checking for legacy files...")
    convert_from_legacy(dataset)

    # Generators
    if not show_plot:
        plt.ioff()

    fig = plt.figure()

    for divergence in divergence_names:
        try:
            with open(f"{dataset}_IMGS/{divergence}/DataDivergenceTraining.txt", "r") as file:
                training = json.load(file)
                file.close
        except:
            print(f"Unable to open {dataset}_IMGS/{divergence}/DataDivergenceTraining.txt; skipping entry.\nFile may be displaced or deleted")
            continue

        epochs = [i for i in range(len(training['gen_loss']))]

        plt.plot(epochs, training['gen_loss'], label=divergence, color=divergence_colors[divergence])

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Evolution of All Generator Losses\n({dataset} Trained With Respective Divergences)')

    plt.savefig(f"{dataset}_IMGS/PlotDivergenceAllGenerator.png")

    if not show_plot:
        plt.close(fig)
    else:
        plt.show()

    # Discriminators
    if not show_plot:
        plt.ioff()

    fig = plt.figure()
    
    for divergence in divergence_names:
        try:
            with open(f"{dataset}_IMGS/{divergence}/DataDivergenceTraining.txt", "r") as file:
                training = json.load(file)
                file.close
        except:
            print(f"Unable to open {dataset}_IMGS/{divergence}/DataDivergenceTraining.txt; skipping entry.\nFile may be displaced or deleted")
            continue

        epochs = [i for i in range(len(training['gen_loss']))]

        plt.plot(epochs, training['gen_loss'], label=divergence, color=divergence_colors[divergence])

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Evolution of All Discriminator Losses\n({dataset} Trained With Respective Divergences)')

    plt.savefig(f"{dataset}_IMGS/PlotDivergenceAllDiscriminator.png")

    if not show_plot:
        plt.close(fig)
    else:
        plt.show()


# Plots the evolution of real / fake statistics for a given training divergence
# This data is generated from a single run, trained on a single divergence
def plot_real_fake_training(dataset, divergence, show_plot=True):
    try:
        with open(f"{dataset}_IMGS/{divergence}/DataDivergenceTraining.txt", "r") as file:
            training = json.load(file)
            file.close
    except:
        print(f"Unable to open {dataset}_IMGS/{divergence}/DataDivergenceTraining.txt.\nSearching for a legacy format Losses.txt to convert...")
        try:
            convert_from_legacy(dataset, divergence=divergence)
            print(f"Conversion succeeded: created file {dataset}_IMGS/{divergence}/DataDivergenceTraining.txt. Retrying...")
        except:
            print("Conversion failed.")
            return
        # Recursive retry
        return plot_real_fake_training(dataset, divergence, show_plot=show_plot)

    if (not 'real_stat' in training) or (not 'fake_stat' in training):
        print(f"Data {dataset}_IMGS/{divergence}/DataDivergenceTraining.txt contains no real / fake statistics.\nIf you converted this data from a legacy format, this likely means the legacy format does not contain the real / fake statistics: you will need to run a training session again.")
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

    plt.savefig(f"{dataset}_IMGS/{divergence}/PlotStatTrainingRealFake.png")

    if not show_plot:
        plt.close(fig)
    else:
        plt.show()

# Plots the evolution of real / fake statistics for all training divergences
# This data is generated from multiple runs, each trained with its own divergence
def plot_real_fake_all(dataset, show_plot=True):
    print("Checking for legacy files...")
    convert_from_legacy(dataset)

    # Real statistic
    if not show_plot:
        plt.ioff()

    fig = plt.figure()
    
    for divergence in divergence_names:
        try:
            with open(f"{dataset}_IMGS/{divergence}/DataDivergenceTraining.txt", "r") as file:
                training = json.load(file)
                file.close
        except:
            print(f"Unable to open {dataset}_IMGS/{divergence}/DataDivergenceTraining.txt; skipping entry.\nFile may be displaced or deleted.")
            continue

        if (not 'real_stat' in training) or (not 'fake_stat' in training):
            print(f"Data {dataset}_IMGS/{divergence}/DataDivergenceTraining.txt contains no real / fake statistics; skipping entry.\nIf you converted this data from a legacy format, this likely means the legacy format does not contain the real / fake statistics: you will need to run a training session again.")
            continue

        epochs = [i for i in range(len(training['real_stat']))]

        plt.plot(epochs, training['real_stat'], label=divergence, color=divergence_colors[divergence])

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Evolution of Real Statistics\n({dataset} Trained With Respective Divergences)')

    plt.savefig(f"{dataset}_IMGS/PlotStatAllReal.png")

    if not show_plot:
        plt.close(fig)
    else:
        plt.show()

    # Fake statistic
    if not show_plot:
        plt.ioff()

    fig = plt.figure()

    for divergence in divergence_names:
        try:
            with open(f"{dataset}_IMGS/{divergence}/DataDivergenceTraining.txt", "r") as file:
                training = json.load(file)
                file.close
        except:
            print(f"Unable to open {dataset}_IMGS/{divergence}/DataDivergenceTraining.txt; skipping entry.\nFile may be displaced or deleted.")
            continue

        if (not 'real_stat' in training) or (not 'fake_stat' in training):
            print(f"Data {dataset}_IMGS/{divergence}/DataDivergenceTraining.txt contains no real / fake statistics; skipping entry.\nIf you converted this data from a legacy format, this likely means the legacy format does not contain the real / fake statistics: you will need to run a training session again.")
            continue

        epochs = [i for i in range(len(training['fake_stat']))]

        plt.plot(epochs, training['fake_stat'], label=divergence, color=divergence_colors[divergence])

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Evolution of Fake Statistics\n({dataset} Trained With Respective Divergences)')

    plt.savefig(f"{dataset}_IMGS/PlotStatAllFake.png")

    if not show_plot:
        plt.close(fig)
    else:
        plt.show()

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