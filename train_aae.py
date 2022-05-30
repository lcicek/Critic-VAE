from tabnanny import verbose
import torch
import torch.nn.functional as F
from torch import Tensor
import time
import os
import minerl
import numpy as np
from time import time

from logger import Logger
from nets import *
from parameters import *
from utility import *

logger = Logger('./logs/aae' + str(time())[-5::])

### Initialize mineRL dataset ###
os.environ['MINERL_DATA_ROOT'] = MINERL_DATA_ROOT_PATH
data = minerl.data.make('MineRLTreechop-v0', num_workers=1)
all_pov_obs = load_minerl_data(data) # get all minerl observations, divide in 2 for lower memory usage
del data # without this line errors gets thrown at the end of the program

### Load trained critic model ###
print('loading critic...')
critic = Critic().to(torch.device('cpu'))
critic.load_state_dict(torch.load(CRITIC_PATH, map_location='cpu'))

### Preprocess minerl data; Divide evenly into high/low-value images ###
dset = prepare_data(all_pov_obs, critic)

### Initialize networks ###
Q = Q_net(X_dim=n_channels, N=n, z_dim=z_dim).to(device)
P = P_net(X_dim=n_channels, N=n, z_dim=z_dim).to(device)
D = D_net(z_dim, z_dim).to(device)

with torch.no_grad():
    Q.eval()
    _, z_sample = Q(torch.zeros((BATCH_SIZE, n_channels, h, h)).to(device))
    _ = D(z_sample)
    Q.train()

optim_P = torch.optim.Adam(P.parameters(), lr=0.0001)
optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=0.0001)

#regularizing optimizers
optimizer_class = torch.optim.SGD
optim_Q_gen = optimizer_class(Q.parameters(), lr=0.001) # Generator
optim_D = optimizer_class(D.parameters(), lr=0.05, momentum=0.25) # Discriminator classification

scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optim_Q_gen, milestones=[5], gamma=4, verbose=True)
scheduler2 = torch.optim.lr_scheduler.StepLR(optim_D, step_size=5, gamma=0)

one_label = torch.ones((BATCH_SIZE, 1), device='cuda')
zero_label = torch.zeros((BATCH_SIZE, 1), device='cuda')

num_samples = dset.shape[0]

# Start training
for ep in range(EPOCHS):
    epoch_indices = np.arange(num_samples)
    np.random.shuffle(epoch_indices)

    print(f'epoch: {ep}')

    for batch_i in range(0, num_samples, BATCH_SIZE):
        # NOTE: this will cut off incomplete batches from end of the random indices
        batch_indices = epoch_indices[batch_i:batch_i + BATCH_SIZE]

        all_data = dset[batch_indices]
        
        images = np.array([d[0] for d in all_data])
        images = Tensor(images).to(device)
        labels = Tensor(np.array([d[1] for d in all_data])).to(device)
        classes = F.one_hot(labels.long(), num_classes=NUM_CLASSES)

        optim_P.zero_grad()
        optim_Q_enc.zero_grad()
        optim_Q_gen.zero_grad()
        optim_D.zero_grad()

        ### Autoencoder and Classifier ###
        class_out, z_sample = Q(images)
        X_sample = P(z_sample)

        recon_loss = F.mse_loss(X_sample, images)
        c_loss = F.binary_cross_entropy(class_out, classes.float())

        total_loss = recon_loss + c_loss
        total_loss.backward()
            
        optim_P.step()
        optim_Q_enc.step()

        ### GENERATOR ###
        _, z_sample = Q(images)
        D_fake = D(z_sample)
        
        # Generator loss
        G_loss = F.binary_cross_entropy_with_logits(D_fake, one_label)
        G_loss.backward()
        optim_Q_gen.step()
    
        ### DISCRIMINATOR ###
        D_fake = D(z_sample.detach())
        z_real = sample_gauss(labels)
        D_real = D(z_real)

        #Discriminator classification loss
        real_loss =  F.binary_cross_entropy(D_real, one_label)
        fake_loss = F.binary_cross_entropy(D_fake, zero_label)
        D_loss = real_loss + fake_loss

        D_loss.backward()
        optim_D.step()

        #============ TensorBoard logging ============# 
        # Log after each epoch
        # (1) Log the scalar values
        if batch_i % collect == 0:
            print(f'step {batch_i + (DATA_SAMPLES * ep)}')

            info = {
                'recon_loss': recon_loss.item(),
                'classifier_loss': c_loss.item(),
                'discriminator_real_loss': real_loss.item(),
                'discriminator_fake_loss': fake_loss.item(),
                'generator_loss': G_loss.item(),
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, batch_i + (DATA_SAMPLES * ep))

    scheduler1.step()
    scheduler2.step()

# Save states
torch.save(Q.state_dict(),'Q_encoder_weights.pt')
torch.save(P.state_dict(),'P_decoder_weights.pt')
torch.save(D.state_dict(),'D_discriminator_weights.pt')