from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from PIL import Image
import torch
import json
import numpy as np
import matplotlib.pyplot as plt

div=['total_variation', 'forward_kl', 'hellinger', 'jensen_shannon', 'pearson', 'reverse_kl', 'piecewise']
colors={'total_variation':'blue', 'forward_kl':'orange', 'hellinger':'pink', 'jensen_shannon':'gray', 'pearson':'green', 'reverse_kl':'purple', 'piecewise':'black'}
CleanName={'total_variation':'Total Variation', 'forward_kl':'Forward KL', 'hellinger':'Hellinger', 'jensen_shannon':'Jensen-Shannon', 'pearson':'Pearson', 'reverse_kl':'Reverse KL', 'piecewise':'Piecewise'}
#Redo hellinger
# div=['total_variation']

folder='experiments/MNIST'
suffixe='Pretty_three'

mural=None

for dd in div:
    for run in range(50, -1, -1):
        try:
            ff=json.load(open(f'{folder}/{dd}/{run:0>3}/DataDivergenceOther.txt', 'r'))
            fig = plt.figure()
            plt.ioff()
            for elem in ff:
                y=[el*-1 for el in elem['dis_loss']]
                x=np.arange(len(y))
                plt.plot(x, y, label=CleanName[elem['divergence']], color=colors[elem['divergence']])

            #Tracing divergence
            ff = json.load(open(f'{folder}/{dd}/{run:0>3}/DataDivergenceTraining.txt', 'r'))
            y = [f*-1 for f in ff['dis_loss']]
            x = np.arange(len(y))
            plt.plot(x, y, label=CleanName[ff['divergence']], color=colors[ff['divergence']])

            plt.xlabel('Epoch')
            plt.ylabel('Estimated Lower Bound')
            plt.legend()
            axes = plt.gca()
            # axes.set_xlim([xmin,xmax])
            axes.set_ylim([-10, 10])
            # plt.title(f'Lower bound over all divergence when minimizing \n the {ff["divergence"]} divergence')
            plt.savefig(f"Graphes_Minim_Divergences/{dd}.png")
            # plt.savefig(f"test.png")
            plt.close(fig)
            print("hwllo")
            break
        except Exception as e:
            print(e)
            continue