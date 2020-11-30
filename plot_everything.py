import sys
import os
import matplotlib.pyplot as plt
import ast

suffixe='new_update'

# Choose dataset
for dataset in ['MNIST']:#,'CIFAR','Gaussian','svhn']:
    for is_gen in [0,1]:
        fig = plt.figure()
        plt.ioff()
        for loss,color in zip(['total_variation', 'forward_kl', 'reverse_kl', 'pearson', 'hellinger', 'jensen_shannon'],['blue','orange','purple','green','pink','gray']):
            #If there is no file with a specific loss, just skip it     
            try: file = open(f"{dataset}_IMGS/{loss}/Losses.txt", "r")
            except:
                print('Skipping loss '+str(loss)+' for dataset '+str(dataset))
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