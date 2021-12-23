import torch
import numpy as np
from tqdm.notebook import trange, tqdm

from config import GAN_Config
from utils import apply_noise, save_fake_images, save_models


def train_GAN(G, D, data_loader, loss_fn, config: GAN_Config):
    D_optim = torch.optim.Adam(D.parameters(), betas=(0.5, 0.999), lr=config.dlr, weight_decay=1e-4, amsgrad=True)
    G_optim = torch.optim.Adam(G.parameters(), betas=(0.5, 0.999), lr=config.glr, weight_decay=1e-4, amsgrad=True)

    def train_generator():
        # Generate fake images and calculate loss
        z = torch.Tensor(np.random.normal(0, 1, (config.batch_size, config.latent_size))).to(config.device)

        # Add noise to the noisy input.
        # This helps our model better cover the input space and generate more diverse samples
        if config.use_noise is True:
            z = apply_noise(z, config.device)

        fake_images = G(z)
        # calculate the generator loss
        labels = (torch.ones(config.batch_size, 1) * config.label_smooth).to(config.device)
        outputs = D(fake_images)
        g_loss = loss_fn(outputs, labels)

        # Backprop and optimize
        g_loss.backward()
        G_optim.step()

        # Reset gradients
        G_optim.zero_grad()
        D_optim.zero_grad()
        return g_loss, fake_images

    def train_discriminator(images):
        # Create the labels which are later used as input for the BCE loss
        real_labels = (torch.ones(config.batch_size, 1) * config.label_smooth).to(config.device)
        fake_labels = (torch.ones(config.batch_size, 1) * (1 - config.label_smooth)).to(config.device)

        # Add noise to discriminator inputs (both the real and synthetic data) to discourage it from being overconfident
        # about its classification, or relying on a limited set of features to distinguish between training data and
        # generator’s output
        if config.use_noise is True:
            images = apply_noise(images, config.device)

        outputs = D(images)
        # Loss for real images
        loss_r = loss_fn(outputs, real_labels)
        real_score = outputs

        # Loss for fake images
        z = torch.randn(config.batch_size, config.latent_size).to(config.device)
        if config.use_noise is True:
            z = apply_noise(z, config.device)
        fake_images = G(z)

        if config.use_noise is True:
            fake_images = apply_noise(fake_images, config.device)

        outputs = D(fake_images)
        loss_f = loss_fn(outputs, fake_labels)
        fake_score = outputs
        # Sum losses
        d_loss = loss_r + loss_f
        # Adjust the parameters using backprop
        d_loss.backward()
        # Compute gradients
        D_optim.step()
        # Reset gradients
        G_optim.zero_grad()
        D_optim.zero_grad()

        return d_loss, real_score, fake_score

    total_step = len(data_loader)
    d_losses, g_losses, real_scores, fake_scores = [], [], [], []
    G.to(config.device)
    D.to(config.device)

    bar_1 = trange(config.num_epochs, desc='Epochs')
    for epoch in bar_1:
        bar_1.set_description(f"Epoch [{epoch + 1:02d}/{config.num_epochs:02d}]")
        bar_2 = tqdm(enumerate(data_loader), desc=f'Running {len(data_loader):03d} mini batches', leave=False)

        G.train()
        D.train()

        still_training = True
        for i, (images, _) in bar_2:
            # Load a batch & transform to vectors
            images = images.to(config.device)
            # Train the generator n times
            for _ in range(config.generator_speedup):
                g_loss, fake_images = train_generator()
            # Train the discriminator  once
            d_loss, real_score, fake_score = train_discriminator(images)
            # Inspect the losses
            bar_2.set_description(
                "Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}\n".format(
                    i + 1, total_step, d_loss.item(), g_loss.item(), real_score.mean().item(),
                    fake_score.mean().item()))
            if (i + 1) % 200 == 0:
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

                real_scores.append(real_score.mean().item())
                fake_scores.append(fake_score.mean().item())
                if real_score.mean().item() == 1.0:
                    still_training = False

        if not still_training:
            break

        # Sample and save images
        save_fake_images(G, config, index=epoch + 1)
        save_models(D, G, config, f"{config.exp_name}_epoch#{epoch:03d}")

    return d_losses, g_losses, real_scores, fake_scores
