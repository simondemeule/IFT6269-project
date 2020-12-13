from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from PIL import Image
import torch

div=['total_variation', 'forward_kl', 'hellinger', 'jensen_shannon', 'pearson', 'reverse_kl', 'piecewise']
# div=['forward_kl', 'hellinger', 'pearson', 'reverse_kl']

folder='experiments/MNIST'
suffixe='Pretty_three'

mural=None

for dd in div:
    for i in [0,5,49]:
        for run in range(50, -1, -1):
            try:
                ff=Image.open(f'{folder}/{dd}/{run:0>3}/GRID{i}.png')
                ff=TF.to_tensor(ff).unsqueeze(0)
                if mural is None:
                    mural=ff
                else:
                    mural=torch.cat([mural, ff], axis=0)
                break
            except:
                continue


save_image(mural.view(-1, 3, 152, 152), f"experiments/MNIST/ComparisonOfDivergence{suffixe}.png", nrow=3)