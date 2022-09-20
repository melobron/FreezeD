import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

import argparse
import random
import numpy as np
import math
import cv2

from models.StyleGAN2 import StyledGenerator

# Arguments
parser = argparse.ArgumentParser(description='Test StyleGAN')

parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--model_path', default='./pre-trained/stylegan-256px-new.model', type=str)
parser.add_argument('--dataset_name', default='FFHQ', type=str)
parser.add_argument('--img_size', default=256, type=int)  # Pre-trained model suited for 256

# Mean Style
parser.add_argument('--style_mean_num', default=10, type=int)  # Style mean calculation for Truncation trick

# Sample Generation
parser.add_argument('--n_row', default=3, type=int)  # For Visualization
parser.add_argument('--n_col', default=5, type=int)  # For Visualization
parser.add_argument('--alpha', default=1, type=float)  # ?
parser.add_argument('--style_weight', default=0.7, type=float)  # 0: Mean of FFHQ, 1: Independent

# Style Mixing
parser.add_argument('--n_samples', default=5, type=int)

# Transformations
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=tuple, default=(0.5, 0.5, 0.5))
parser.add_argument('--std', type=tuple, default=(0.5, 0.5, 0.5))

opt = parser.parse_args()


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


@torch.no_grad()
def generate_samples(generator, device, n_samples, step, alpha, mean_style, style_weight):
    imgs = generator(torch.randn(n_samples, 512).to(device), step=step, alpha=alpha,
                     mean_style=mean_style, style_weight=style_weight)
    return imgs


def visualize_samples(imgs, rows, cols, mean, std):
    # Tile batch images
    b, c, h, w = imgs.shape
    tile = torch.zeros(size=(c, h*rows, w*cols))
    for i in range(rows):
        for j in range(cols):
            index = i*rows + j
            start_h, start_w = h*i, w*j
            tile[:, start_h:start_h+h, start_w:start_w+w] = imgs[index]

    # Visualization
    transform = transforms.Compose([
        transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
    ])
    tile = transform(tile)
    tile = tile.cpu().numpy().transpose(1, 2, 0)
    tile = np.clip(tile, 0., 1.) * 255.
    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./results/sample.png', tile)


# @torch.no_grad()
# def style_mixing():
#
#     # Generate Random Samples
#     # img = sample(generator, step, mean_style, args.n_row * args.n_col, device)
#     img = generator()


if __name__ == "__main__":
    device = torch.device('cuda:{}'.format(opt.gpu_num))

    # Random Seeds
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    # Model
    generator = StyledGenerator().to(device)
    generator.load_state_dict(torch.load(opt.model_path)['g_running'])
    generator.eval()

    # Mean Styles
    mean_style = get_mean_style(generator, device, style_mean_num=opt.style_mean_num)

    # Parameters
    step = int(math.log(opt.img_size, 2)) - 2

    # 1. Generate Samples
    imgs = generate_samples(generator, device, n_samples=opt.n_row*opt.n_col, step=step, alpha=opt.alpha,
                            mean_style=mean_style, style_weight=opt.style_weight)
    visualize_samples(imgs, rows=opt.n_row, cols=opt.n_col, mean=opt.mean, std=opt.std)

    # 2. Style Mixing
