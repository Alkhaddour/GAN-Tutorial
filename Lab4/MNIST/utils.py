import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from torchvision.utils import save_image
from config import GAN_Config


# denormalization image from range (-1)-1 to range 0-1 to display it
def batch_denorm(x, image_c, image_h, image_w):
    batch_size_ = x.size(0)
    x = x.view(batch_size_, -1)
    x -= x.min(1, keepdim=True)[0]
    x /= x.max(1, keepdim=True)[0]
    x = x.view(batch_size_, image_c, image_h, image_w)
    return x


# show image sample with matplotlib
def plot_torch_image(img, **kwargs):
    """
    Input image is a torch tensor with the following dims (C,H,W)
    To plot it with matplotlib, we need to change it to (H,W,C)
    kwargs variable is used to pass other parameters to 'imshow' function.
    """
    plt.imshow(img.permute(1, 2, 0), **kwargs)


# weight initializer
def init_weights(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def apply_noise(z, device, coef=0.005):
    noise = coef * np.random.uniform() * torch.amax(z)
    z = z + (noise.to(device) * torch.randn(size=z.shape).to(device))
    return z


def save_fake_images(G, config: GAN_Config, index):
    G.eval()
    sample_vectors = torch.Tensor(np.random.normal(0, 1, (config.n_samples,
                                                          config.latent_size))).to(config.device)
    sample_vectors = apply_noise(sample_vectors, config.device)

    fake_images = G(sample_vectors)
    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    save_image(batch_denorm(fake_images, config.image_c, config.image_h, config.image_w),
               os.path.join(config.samples_dir, fake_fname),
               nrow=int(np.sqrt(config.n_samples)))


def save_models(D, G, config: GAN_Config, suffix):
    torch.save(
        G.state_dict(),
        f"{config.models_dir}/{suffix}_generator_model.pth")
    torch.save(
        D.state_dict(),
        f"{config.models_dir}/{suffix}_discriminator_model.pth")


def plot_losses(d_losses, g_losses, plt_title):
    plt.figure()
    plt.plot(d_losses, '-')
    plt.plot(g_losses, '-')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend(['Discriminator', 'Generator'])
    plt.title(f"Losses -- {plt_title}");


def plot_scores(real_scores, fake_scores, plt_title):
    plt.figure()
    plt.plot(real_scores, '-')
    plt.plot(fake_scores, '-')
    plt.xlabel('Step')
    plt.ylabel('Score')
    plt.yticks([x * 0.1 for x in range(0, 11)])
    plt.legend(['Real score', 'Fake score'])
    plt.title(f"Scores -- {plt_title}");