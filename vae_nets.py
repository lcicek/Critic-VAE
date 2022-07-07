import torch
import torch.nn.functional as F
from torch import nn, Tensor
from vae_parameters import *
from math import exp

class VariationalAutoencoder(nn.Module):
    def __init__(self, dims=[32, 64, 128, 256]):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(dims)
        self.decoder = Decoder(dims)
        self.mssim_loss = MSSIM()

    def forward(self, x, pred):
        mu, logvar = self.encoder(x)
        z_sample = self.reparametrize(mu, logvar)
        recon = self.decoder(z_sample, pred)

        return x, mu, logvar, recon
    
    def recon_samples(self, x, reward):
        mu, logvar = self.encoder(x)
        recons = []
        for _ in range(6):
            sample = self.reparametrize(mu, logvar)
            recon = self.decoder(sample, reward)
            recons.append(recon)
        
        return recons

    def inject(self, x, reward=Tensor([0, 0.2, 0.4, 0.6, 0.8, 1])):
        reward = reward.to(device)
        mu, _ = self.encoder(x)

        recons = []
        for i in range(inject_n):
            recon = self.decoder(mu, reward[i].view(1), evalu=True)
            recons.append(recon)
        
        return recons

    def evaluate(self, x, pred):
        mu, _ = self.encoder(x)
        recon = self.decoder(mu, pred.view(1), evalu=True)

        return recon

    def reparametrize(self, mu, logvar): # logvar is variance
        std = torch.exp(0.5 * logvar) # variance**2 = std
        eps = torch.randn_like(std)
        return mu + eps * std # mean + random * standard-deviation

    def vae_loss(self, x, mu, logvar, recon):        
        #recon_loss = F.mse_loss(recon, x)
        torch.cuda.empty_cache()
        recon_loss = self.mssim_loss(recon, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        
        loss = recon_loss + kld_loss * kld_weight

        return {'total_loss': loss, 'recon_loss':recon_loss.detach(), 'KLD':-kld_loss.detach()}

class VariationalEncoder(nn.Module):
    def __init__(self, dims):
        super(VariationalEncoder, self).__init__()

        self.model = nn.Sequential(
                        nn.Conv2d(ch, dims[0], k, step, p), # to 64x64x32
                        nn.BatchNorm2d(dims[0]),
                        nn.MaxPool2d(2), # to 32x32x32
                        nn.ReLU(),

                        nn.Conv2d(dims[0], dims[1], k, step, p), # to 32x32x64
                        nn.BatchNorm2d(dims[1]),
                        nn.MaxPool2d(2), # to 16x16x64
                        nn.ReLU(),
                        
                        nn.Conv2d(dims[1], dims[2], k, step, p), # to 16x16x128
                        nn.BatchNorm2d(dims[2]),
                        nn.MaxPool2d(2), # to 8x8x128
                        nn.ReLU(),
                        
                        nn.Conv2d(dims[2], dims[3], k, step, p), # to 8x8x256
                        nn.BatchNorm2d(dims[3]),
                        nn.MaxPool2d(2), # to 4x4x256
                        nn.ReLU(),
                    )

        self.fcs = nn.Sequential(
            nn.Linear(bottleneck, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, bottleneck),
            nn.ReLU()
        )

        # mu = mean, sigma = var; "fc" = fully connected layer
        self.fc_mu = nn.Linear(bottleneck, latent_dim)
        self.fc_var = nn.Linear(bottleneck, latent_dim)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)

        z_flat = torch.flatten(x, start_dim=1)
        z_flat = self.fcs(z_flat)

        mu = self.fc_mu(z_flat)
        log_var = self.fc_var(z_flat)
        
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
                        nn.Conv2d(dims[3], dims[2], k, step, p),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2),
                        
                        nn.Conv2d(dims[2], dims[1], k, step, p),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2),
                        
                        nn.Conv2d(dims[1], dims[0], k, step, p),
                        nn.ReLU(),  
                        nn.Upsample(scale_factor=2),

                        nn.Conv2d(dims[0], dims[0], k, step, p),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2),
                        
                        nn.Conv2d(dims[0], ch, k, step, p),
                        nn.Tanh() # tanh-range is [-1, 1], sigmoid is [0, 1]
                    )

        self.decoder_input = nn.Linear(latent_dim+1, bottleneck)

    def forward(self, z, pred, evalu=False, dim=1):
        if evalu:
            z = z[0] # batch_size is 1 when evaluating
            dim = 0
        X = self.decoder_input(torch.cat((z, pred), dim=dim))
        X = X.view(-1, 256, 4, 4)
        X = self.model(X)

        return X

# from: https://github.com/AntixK/PyTorch-VAE
class MSSIM(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 window_size: int=11,
                 size_average:bool = True) -> None:
        """
        Computes the differentiable MS-SSIM loss
        Reference:
        [1] https://github.com/jorge-pessoa/pytorch-msssim/blob/dev/pytorch_msssim/__init__.py
            (MIT License)
        :param in_channels: (Int)
        :param window_size: (Int)
        :param size_average: (Bool)
        """
        super(MSSIM, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.size_average = size_average

    def gaussian_window(self, window_size:int, sigma: float) -> Tensor:
        kernel = torch.tensor([exp((x - window_size // 2)**2/(2 * sigma ** 2))
                               for x in range(window_size)])
        return kernel/kernel.sum()

    def create_window(self, window_size, in_channels):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(in_channels, 1, window_size, window_size).contiguous()
        return window

    def ssim(self,
             img1: Tensor,
             img2: Tensor,
             window_size: int,
             in_channel: int,
             size_average: bool) -> Tensor:

        device = img1.device
        window = self.create_window(window_size, in_channel).to(device)
        mu1 = F.conv2d(img1, window, padding= window_size//2, groups=in_channel)
        mu2 = F.conv2d(img2, window, padding= window_size//2, groups=in_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding = window_size//2, groups=in_channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding = window_size//2, groups=in_channel) - mu2_sq
        sigma12   = F.conv2d(img1 * img2, window, padding = window_size//2, groups=in_channel) - mu1_mu2

        img_range = 1.0 #img1.max() - img1.min() # Dynamic range
        C1 = (0.01 * img_range) ** 2
        C2 = (0.03 * img_range) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)
        return ret, cs

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        device = img1.device
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        mssim = []
        mcs = []

        for _ in range(levels):
            sim, cs = self.ssim(img1, img2,
                                self.window_size,
                                self.in_channels,
                                self.size_average)
            mssim.append(sim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        # # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
        # if normalize:
        #     mssim = (mssim + 1) / 2
        #     mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = mssim ** weights

        output = torch.prod(pow1[:-1] * pow2[-1])
        return 1 - output