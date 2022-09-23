import torch
import torchvision.transforms as transforms
import math
import numpy as np
from models.StyleGAN2 import StyledGenerator
import matplotlib.pyplot as plt
import cv2


@torch.no_grad()
def get_mean_style(generator, device, style_mean_num):
    mean_style = None

    for _ in range(style_mean_num):
        style = generator.mean_style(torch.randn(1024, 512).to(device))
        if mean_style is None:
            mean_style = style
        else:
            mean_style += style

    mean_style /= style_mean_num
    return mean_style

gpu_num = 0
device = torch.device('cuda:{}'.format(gpu_num))

generator = StyledGenerator().to(device)
# generator.load_state_dict(torch.load(opt.model_path)['g_running'])
generator.load_state_dict(torch.load('{}'.format('./experiments/exp1/checkpoints/G_200epochs.pth'), map_location=device))
generator.eval()

n_source, n_target = 5, 3
source_code = torch.randn(n_source, 512).to(device)
target_code = torch.randn(n_target, 512).to(device)

mean_style = get_mean_style(generator, device, style_mean_num=10)

i = 0
step = int(math.log(256, 2)) - 2
alpha = 1
style_weight = 0.7
with torch.no_grad():
    mixed_imgs = generator([target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
                           step=step, alpha=alpha, mean_style=mean_style, style_weight=style_weight,
                           mixing_range=(0, 1))

    source_imgs = generator(source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=style_weight)
    target_imgs = generator(target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=style_weight)

# Visualization
mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
transform = transforms.Compose([
    transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std])
])

fig = plt.figure(figsize=(15, 10))
rows, cols = 3, 5
tile = transform(source_imgs[0])
tile = tile.cpu().numpy().transpose(1, 2, 0)
tile = np.clip(tile, 0., 1.)
tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
plt.subplot(rows, cols, 1)
plt.imshow(tile)
plt.title('source')
tile = transform(target_imgs[0])
tile = tile.cpu().numpy().transpose(1, 2, 0)
tile = np.clip(tile, 0., 1.)
tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
plt.subplot(rows, cols, 2)
plt.imshow(tile)
plt.title('target')

for i in range(5):
    tile = transform(source_imgs[i])
    tile = tile.cpu().numpy().transpose(1, 2, 0)
    tile = np.clip(tile, 0., 1.)
    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
    plt.subplot(rows, cols, i+6)
    plt.imshow(tile)

for i in range(5):
    tile = transform(mixed_imgs[i])
    tile = tile.cpu().numpy().transpose(1, 2, 0)
    tile = np.clip(tile, 0., 1.)
    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
    plt.subplot(rows, cols, i+11)
    plt.imshow(tile)
plt.show()

