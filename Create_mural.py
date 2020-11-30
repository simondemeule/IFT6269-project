from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from PIL import Image
import torch

div=['total_variation', 'forward_kl', 'hellinger', 'jensen_shannon', 'pearson', 'reverse_kl']
# div=['forward_kl', 'hellinger', 'pearson', 'reverse_kl']

folder='MNIST_IMGS'
suffixe='NewUpdate'

mural=None

for dd in div:
    for i in range(0, 50, 5):
        ff=Image.open(f'{folder}/{dd}/GRID{i}.png')
        ff=TF.to_tensor(ff).unsqueeze(0)
        if mural is None:
            mural=ff
        else:
            mural=torch.cat([mural, ff], axis=0)


save_image(mural.view(-1, 3, 152, 152), f"MNIST_IMGS/ComparisonOfDivergence{suffixe}.png", nrow=10)