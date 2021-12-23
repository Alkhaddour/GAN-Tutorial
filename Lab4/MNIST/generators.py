import torch
import torch.nn as nn

from normalization import SpectralNorm


class SimpleGenerator(nn.Module):
    def __init__(self, nc, h, w, latent_size, ngf):
        super(SimpleGenerator, self).__init__()
        self.nc = nc
        self.h = h
        self.w = w
        self.ngf = ngf

        # Projection from latent space to flattened image
        self.linear = nn.Linear(latent_size, nc * h * w)

        self.deconv_layers = nn.Sequential(
            # First deconv layer
            nn.ConvTranspose2d(in_channels=nc, out_channels=ngf, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),

            # Second deconv layer
            nn.ConvTranspose2d(in_channels=ngf, out_channels=ngf, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),

            # Third deconv layer
            nn.ConvTranspose2d(in_channels=ngf, out_channels=ngf, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),

            # Fourth deconv layer
            nn.ConvTranspose2d(in_channels=ngf, out_channels=ngf, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),

            # Fifth deconv layer
            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=(3, 3), stride=1, padding=1),
            # Not a mistake, no batch BatchNorm here ))
        )

    def forward(self, x):
        bs = x.shape[0]
        x = self.linear(x)
        x = x.view(bs, self.nc, self.h, self.w)
        x = self.deconv_layers(x)
        return torch.tanh(x)


class SN_Generator(nn.Module):
    def __init__(self, nc, h, w, latent_size, ngf):
        super(SN_Generator, self).__init__()
        self.nc = nc
        self.h = h
        self.w = w
        self.ngf = ngf

        # Projection from latent space to flattened image
        self.linear = nn.Linear(latent_size, nc * h * w)

        self.deconv_layers = nn.Sequential(
            # First deconv layer
            SpectralNorm(nn.ConvTranspose2d(in_channels=nc, out_channels=ngf, kernel_size=(3, 3), stride=1, padding=1)),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),

            # Second deconv layer
            SpectralNorm(
                nn.ConvTranspose2d(in_channels=ngf, out_channels=ngf, kernel_size=(3, 3), stride=1, padding=1)),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),

            # Third deconv layer
            SpectralNorm(
                nn.ConvTranspose2d(in_channels=ngf, out_channels=ngf, kernel_size=(3, 3), stride=1, padding=1)),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),

            # Fourth deconv layer
            SpectralNorm(
                nn.ConvTranspose2d(in_channels=ngf, out_channels=ngf, kernel_size=(3, 3), stride=1, padding=1)),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),

            # Fifth deconv layer
            SpectralNorm(nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=(3, 3), stride=1, padding=1)),
            # Not a mistake, no batch BatchNorm here ))
        )

    def forward(self, x):
        bs = x.shape[0]
        x = self.linear(x)
        x = x.view(bs, self.nc, self.h, self.w)
        x = self.deconv_layers(x)
        return torch.tanh(x)
