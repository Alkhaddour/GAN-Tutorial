import torch
import torch.nn as nn

from normalization import SpectralNorm


class SimpleDiscriminator(nn.Module):
    def __init__(self, nc, ndf, slope,do):
        super(SimpleDiscriminator, self).__init__()
        self.nc = nc
        self.do = do
        self.ndf = ndf
        self.slope = slope

        self.conv_layers = nn.Sequential(
            # First Layer
            nn.Conv2d(in_channels=self.nc, out_channels=ndf, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(slope),
            nn.Dropout(do),
            # Not a mistake, no batch BatchNorm here ))

            # Second Layer
            nn.Conv2d(in_channels=ndf, out_channels=ndf, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(slope),
            nn.Dropout(do),

            # Third layer
            nn.Conv2d(in_channels=ndf, out_channels=ndf, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(slope),
            nn.Dropout(do),

            # Fourth layer
            nn.Conv2d(in_channels=ndf, out_channels=ndf, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(slope),
            nn.Dropout(do),

            # Fifth layer
            nn.Conv2d(in_channels=ndf, out_channels=ndf, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(slope),
            nn.Dropout(do),
        )
        self.linear = nn.Linear(50176, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return nn.Sigmoid()(x)


class SN_Discriminator(nn.Module):
    def __init__(self, nc, ndf, slope,do):
        super(SN_Discriminator, self).__init__()
        self.nc = nc
        self.do = do
        self.ndf = ndf
        self.slope = slope

        self.conv_layers = nn.Sequential(
            # First Layer
            SpectralNorm(nn.Conv2d(in_channels=self.nc, out_channels=ndf, kernel_size=(3, 3), stride=1, padding=1)),
            nn.LeakyReLU(slope),
            nn.Dropout(do),
            # Not a mistake, no batch BatchNorm here ))

            # Second Layer
            SpectralNorm(nn.Conv2d(in_channels=ndf, out_channels=ndf, kernel_size=(3, 3), stride=1, padding=1)),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(slope),
            nn.Dropout(do),

            # Third layer
            SpectralNorm(nn.Conv2d(in_channels=ndf, out_channels=ndf, kernel_size=(3, 3), stride=1, padding=1)),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(slope),
            nn.Dropout(do),

            # Fourth layer
            SpectralNorm(nn.Conv2d(in_channels=ndf, out_channels=ndf, kernel_size=(3, 3), stride=1, padding=1)),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(slope),
            nn.Dropout(do),

            # Fifth layer
            SpectralNorm(nn.Conv2d(in_channels=ndf, out_channels=ndf, kernel_size=(3, 3), stride=1, padding=1)),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(slope),
            nn.Dropout(do),
        )
        self.linear = nn.Linear(50176, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return nn.Sigmoid()(x)

