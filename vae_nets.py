import torch
import torch.nn.functional as F
from torch import nn, Tensor
from vae_parameters import *

class VariationalAutoencoder(nn.Module):
    def __init__(self, dims=[32, 64, 128, 256]):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(dims)
        self.decoder = Decoder(dims)

    def forward(self, x, reward):
        mu, logvar = self.encoder(x)
        z_sample = self.reparametrize(mu, logvar)
        recon = self.decoder(z_sample, reward)

        return x, mu, logvar, recon
    
    def recon_samples(self, x, reward):
        mu, logvar = self.encoder(x)
        recons = []
        for _ in range(6):
            sample = self.reparametrize(mu, logvar)
            recon = self.decoder(sample, reward)
            recons.append(recon)
        
        return recons

    def inject(self, x, reward=Tensor([0, 0.2, 0.4, 0.6, 0.8, 1]).to(device)):
        mu, _ = self.encoder(x)

        recons = []
        for i in range(inject_n):
            recon = self.decoder(mu, reward[i].view(1), evalu=True)
            recons.append(recon)
        
        return recons

    def evaluate(self, x, reward):
        mu, _ = self.encoder(x)
        recon = self.decoder(mu, reward.view(1), evalu=True)

        return recon

    def reparametrize(self, mu, logvar): # logvar is variance
        std = torch.exp(0.5 * logvar) # variance**2 = std
        eps = torch.randn_like(std)
        return mu + eps * std # mean + random * standard-deviation

    def vae_loss(self, x, mu, logvar, recon):        
        recon_loss = F.mse_loss(recon, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        
        loss = recon_loss + kld_loss * kld_weight

        return {'total_loss': loss, 'recon_loss':recon_loss.detach(), 'KLD':-kld_loss.detach()}

class VariationalEncoder(nn.Module):
    def __init__(self, dims):
        super(VariationalEncoder, self).__init__()

        self.model = nn.Sequential(
                        nn.Conv2d(ch, dims[0], 5, 1, 2), # to 64x64x32
                        nn.BatchNorm2d(dims[0]),
                        nn.MaxPool2d(2), # to 32x32x32
                        nn.ReLU(),

                        nn.Conv2d(dims[0], dims[1], 5, 1, 2), # to 32x32x64
                        nn.BatchNorm2d(dims[1]),
                        nn.MaxPool2d(2), # to 16x16x64
                        nn.ReLU(),
                        
                        nn.Conv2d(dims[1], dims[2], 5, 1, 2), # to 16x16x128
                        nn.BatchNorm2d(dims[2]),
                        nn.MaxPool2d(2), # to 8x8x128
                        nn.ReLU(),
                        
                        nn.Conv2d(dims[2], dims[3], 5, 1, 2), # to 8x8x256
                        nn.BatchNorm2d(dims[3]),
                        nn.MaxPool2d(2), # to 4x4x256
                        nn.ReLU(),
                    )

        # mu = mean, sigma = var; "fc" = fully connected layer
        self.fc_mu = nn.Linear(bottleneck, latent_dim)
        self.fc_var = nn.Linear(bottleneck, latent_dim)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)

        z_flat = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(z_flat)
        log_var = self.fc_var(z_flat)
        
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
                        nn.Conv2d(dims[3], dims[2], 5, 1, 2),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2),
                        
                        nn.Conv2d(dims[2], dims[1], 5, 1, 2),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2),
                        
                        nn.Conv2d(dims[1], dims[0], 5, 1, 2),
                        nn.ReLU(),  
                        nn.Upsample(scale_factor=2),

                        nn.Conv2d(dims[0], dims[0], 5, 1, 2),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2),
                        
                        nn.Conv2d(dims[0], ch, 5, 1, 2),
                        nn.Tanh() # tanh-range is [-1, 1], sigmoid is [0, 1]
                    )
        
        self.decoder_input = nn.Linear(latent_dim+1, bottleneck)

    def forward(self, z, reward, evalu=False, dim=1):
        if evalu:
            z = z[0] # batch_size is 1 when evaluating
            dim = 0
        X = self.decoder_input(torch.cat((z, reward), dim=dim))
        X = X.view(-1, 256, 4, 4)
        X = self.model(X)

        return X
