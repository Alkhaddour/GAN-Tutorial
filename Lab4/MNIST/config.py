import torch
import os


class GAN_Config:
    def __init__(self, image_w=28, image_h=28, image_c=1, batch_size=100, num_epochs=50,latent_size=100, device='auto',
                 ngf=64, ndf=64, dlr=0.0001, glr=0.0001, leaky_slope=0.2, dropout_rate_d=0.0, label_smooth=1.0,
                 use_noise=False, generator_speedup=2, n_samples=100, samples_dir='./samples', models_dir='./models',
                 exp_name='GAN'):
        # data config
        self.image_w = image_w
        self.image_h = image_h
        self.image_c = image_c
        self.image_size = image_w * image_h * self.image_c  # 28*28 flatten

        # Model config
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.latent_size = latent_size  # input random input vector latent
        self.ngf = ngf  # Size of feature maps in generator
        self.ndf = ndf  # Size of feature maps in discriminator
        self.dlr = dlr
        self.glr = glr
        self.leaky_slope = leaky_slope
        self.dropout_rate_d = dropout_rate_d

        # training config
        self.generator_speedup = generator_speedup
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda' or device == 'cpu':
            self.device = device
        else:
            raise ValueError(f"Unknown device type ({device}), use 'cuda', 'cpu' or 'auto'")

        # tricks
        self.label_smooth = label_smooth
        self.use_noise = use_noise

        # output
        self.exp_name = exp_name
        self.n_samples = n_samples
        self.samples_dir = samples_dir
        self.models_dir = models_dir

        if not os.path.exists(self.samples_dir):
            os.makedirs(self.samples_dir)

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
