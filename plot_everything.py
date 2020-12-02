import sys
import os
import matplotlib.pyplot as plt
import ast

suffixe='new_update'

# Choose dataset
for dataset in ['MNIST']:#,'CIFAR','Gaussian','svhn']:
    # plot losses
    for is_gen in [0,1]:
        fig = plt.figure()
        plt.ioff()
        for loss,color in zip(['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon'],['blue','orange','purple','green','pink','gray']):
            #If there is no file with a specific loss, just skip it     
            try: file = open(f"{dataset}_IMGS/{loss}/Losses.txt", "r")
            except:
                print('Skipping loss of' + str(loss) + ' for dataset ' + str(dataset) + ' : exception opening file')
                continue
            contents = file.read()
            losses_dict = ast.literal_eval(contents)
            file.close()

            epochs = [i for i in range(len(losses_dict['Gen_Loss']))]
            if is_gen==0:
                plt.plot(epochs, losses_dict['Discri_Loss'], label=loss, color=color)
            else:
                plt.plot(epochs, losses_dict['Gen_Loss'], label=loss, color=color)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if is_gen==0:
            plt.title('Evolution of the different Discriminator losses')
            plt.savefig(f"{dataset}_IMGS/All_Discriminator_Losses{suffixe}.png")
        else:
            plt.title('Evolution of the different Generator losses')
            plt.savefig(f"{dataset}_IMGS/All_Generator_Losses{suffixe}.png")
        plt.close(fig)

    # plot real / fake statistics
    for is_real in [0,1]:
        fig = plt.figure()
        plt.ioff()
        for loss,color in zip(['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon'],['blue','orange','purple','green','pink','gray']):
            #If there is no file with a specific loss, just skip it     
            try: file = open(f"{dataset}_IMGS/{loss}/Losses.txt", "r")
            except:
                print('Skipping real / fake statistics of ' + str(loss) + ' for dataset ' + str(dataset) + ' : exception opening file')
                continue
            contents = file.read()
            losses_dict = ast.literal_eval(contents)
            file.close()

            if "real_stat" in losses_dict and "fake_stat" in losses_dict:
                epochs = [i for i in range(len(losses_dict['real_stat']))]
                if is_real==0:
                    plt.plot(epochs, losses_dict['real_stat'], label=loss, color=color)
                else:
                    plt.plot(epochs, losses_dict['fake_stat'], label=loss, color=color)
            else:
                print('Skipping real / fake statistics of ' + str(loss) + ' for dataset ' + str(dataset) + ' : missing statistics data')
        plt.xlabel('Epochs')
        plt.ylabel('Classification Rate')
        plt.legend()
        if is_real==0:
            plt.title('Evolution of the correct classification of real examples')
            plt.savefig(f"{dataset}_IMGS/All_Discriminator_Real_Stat{suffixe}.png")
        else:
            plt.title('Evolution of the correct classification of fake examples')
            plt.savefig(f"{dataset}_IMGS/All_Generator_Fake_Stat{suffixe}.png")
        plt.close(fig)