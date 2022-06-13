# SOURCE: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
import torch #; torch.manual_seed(0)
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import os
import minerl
from PIL import Image, ImageDraw, ImageOps
from time import time
import argparse
import cv2

from nets import Critic
from parameters import *
from utility import load_minerl_data, prepare_data, prepare_rgb_image, to_np
from logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument('-t', action='store_true') # train
parser.add_argument('-i', action='store_true') # show recons of samples
args = parser.parse_args()

TRAIN = args.t
INJECT = args.i

device = 'cuda' if torch.cuda.is_available() else 'cpu'
w = 64 # original image width
epochs = 15
latent_dim = 64
bottleneck = 4096 # 4x4x128; bottleneck of convolutional layers
kld_weight = 0.00005 # https://github.com/AntixK/PyTorch-VAE/issues/11 OR https://github.com/AntixK/PyTorch-VAE/issues/35
diff_threshold = 0.02
diff_factor = 10

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
        for i in range(6):
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
                        nn.Conv2d(3, dims[0], 5, 1, 2), # to 64x64x32
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
                        
                        nn.Conv2d(dims[0], 3, 5, 1, 2),
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

def train(autoencoder, dset):
    opt = torch.optim.Adam(autoencoder.parameters(), lr=0.0001) 
    num_samples = dset.shape[0]

    # Start training
    for ep in range(epochs): # change
        epoch_indices = np.arange(num_samples)
        np.random.shuffle(epoch_indices)

        print(f'epoch: {ep}')

        for batch_i in range(0, num_samples, BATCH_SIZE):
            # NOTE: this will cut off incomplete batches from end of the random indices
            batch_indices = epoch_indices[batch_i:batch_i + BATCH_SIZE]

            all_data = dset[batch_indices]
            
            images = np.array([d[0] for d in all_data])
            images = Tensor(images).to(device)

            preds, _ = critic.evaluate(images)

            opt.zero_grad()
            out = autoencoder(images, preds)
            losses = autoencoder.vae_loss(out[0], out[1], out[2], out[3])
            loss = losses['total_loss']
            loss.backward()
            opt.step()

            if batch_i % log_n == 0:
                print(f'step {batch_i + (DATA_SAMPLES * ep)}')

                info = {
                    'recon_loss': losses['recon_loss'].item(),
                    'kld': losses['KLD'].item(),
                    'total_loss': losses['total_loss'].item()
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, batch_i + (DATA_SAMPLES * ep))

    return autoencoder

def save_images(autoencoder, critic):
    folder = os.listdir(EVAL_IMAGES_PATH)
    for i, img_file in enumerate(folder):
        ### LOAD IMAGES AND PREPROCESS ###
        orig_img = Image.open(f'{EVAL_IMAGES_PATH}/{img_file}')
        img_array = np.array(orig_img).astype(np.float32)
        img_array = img_array.transpose(2, 0, 1) # HWC to CHW for critic
        img_array = img_array[np.newaxis, ...]
        img_array /= 255
        img_tensor = Tensor(img_array).to(device)

        preds, _ = critic.evaluate(img_tensor)

        if INJECT:
            orig_recon = autoencoder.evaluate(img_tensor, preds[0])
            recons = autoencoder.inject(img_tensor)

            conc_h = np.concatenate((
                to_np(img_tensor.view(-1, 3, w, w)[0]),
                to_np(orig_recon.view(-1, 3, w, w)[0]),
                to_np(recons[0].view(-1, 3, w, w)[0]),
                to_np(recons[1].view(-1, 3, w, w)[0]),
                to_np(recons[2].view(-1, 3, w, w)[0]),
                to_np(recons[3].view(-1, 3, w, w)[0]),
                to_np(recons[4].view(-1, 3, w, w)[0]),
                to_np(recons[5].view(-1, 3, w, w)[0]),
            ), axis=2)

            _, img = prepare_rgb_image(conc_h)
        else:
            if preds[0] < 0.6:
                continue
            
            recon_one = autoencoder.evaluate(img_tensor, torch.ones(1).to(device))
            recon_zero = autoencoder.evaluate(img_tensor, torch.zeros(1).to(device) + 0.3)

            recon_one = to_np(recon_one.view(-1, 3, w, w)[0])
            recon_zero = to_np(recon_zero.view(-1, 3, w, w)[0])

            diff = cv2.subtract(recon_zero, recon_one)
            diff = abs(diff)
            diff[diff > diff_threshold] *= diff_factor
            diff[diff > 1] = 1
            #diff[diff < 250] = 0
            # _, diff = cv2.threshold(diff, 255, 255, cv2.THRESH_BINARY)

            _, diff_img = prepare_rgb_image(diff)
            diff_img = ImageOps.grayscale(diff_img)

            conc_h = np.array(np.concatenate((
                to_np(img_tensor.view(-1, 3, w, w)[0]),
                recon_one,
                recon_zero,
            ), axis=2))

            _, conc_img = prepare_rgb_image(conc_h)        
            
            img = Image.new('RGB', (64*4, 64))
            img.paste(conc_img, (0, 0))
            img.paste(diff_img, (64*3, 0))

        if INJECT:
            draw = ImageDraw.Draw(img)
            draw.text((w*1+2, 2), f'{preds[0].item():.1f}', (255,255,255))
        
        img.save(f'{SAVE_PATH}/image-{i:03d}.png', format="png")

### Initialize mineRL dataset ###
os.environ['MINERL_DATA_ROOT'] = MINERL_DATA_ROOT_PATH
data = minerl.data.make('MineRLTreechop-v0', num_workers=1)
all_pov_obs = load_minerl_data(data) # get all minerl observations
del data # without this line, error gets thrown at the end of the program

### Load trained critic model ###
print('loading critic...')
critic = Critic()
critic.load_state_dict(torch.load(CRITIC_PATH, map_location='cpu'))
critic.eval()

### Preprocess minerl data; Divide evenly into high/low-value images ###
dset = prepare_data(all_pov_obs, critic, resize=False)
vae = VariationalAutoencoder().to(device) # GPU

critic = critic.to(device)

if TRAIN: # change for training
    logger = Logger('./logs/vae' + str(time())[-5::])
    vae = train(vae, dset)

    torch.save(vae.encoder.state_dict(), 'vae_encoder_weights.pt')
    torch.save(vae.decoder.state_dict(), 'vae_decoder_weights.pt')
else:
    try:
        vae.encoder.load_state_dict(torch.load('saved-networks/vae_encoder_weights.pt'))
        vae.decoder.load_state_dict(torch.load('saved-networks/vae_decoder_weights.pt'))
    except Exception as e:
        print(e)
    
    vae.eval()
    vae.encoder.eval()
    vae.decoder.eval()

    save_images(vae, critic)