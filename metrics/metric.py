import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from fid_score import calculate_frechet_distance


def get_fake_imgs_acts(device, inception, G, step, alpha=1, code_size=512, sample_num=5000, batch_size=16):
    dataset = TensorDataset(torch.randn(sample_num, code_size))
    dataloader = DataLoader(dataset, batch_size=batch_size)

    imgs, acts = [], []
    for gen_in in dataloader:
        gen_in = gen_in.to(device)
        with torch.no_grad():
            fake_imgs = G(gen_in, step=step, alpha=alpha)
            out = inception(fake_imgs)
            

    return imgs, acts


def get_real_images_acts():
    return


def compute_fid(real_acts, fake_acts):
    mu1, sigma1 = (np.mean(real_acts, axis=0), np.cov(real_acts, rowvar=False))
    mu2, sigma2 = (np.mean(fake_acts, axis=0), np.cov(fake_acts, rowvar=False))

    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid
