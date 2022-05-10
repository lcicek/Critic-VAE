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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = Logger('./logs/aae' + str(time())[-5::])

### Initialize mineRL dataset ###
os.environ['MINERL_DATA_ROOT'] = MINERL_DATA_ROOT_PATH
data = minerl.data.make('MineRLTreechop-v0', num_workers=1)
all_obs = load_minerl_data(data) # get all minerl observations
del data

### Load trained critic model ###
critic = Critic().to(device)
critic.load_state_dict(torch.load(CRITIC_PATH, map_location=device))

### Preprocess minerl data; Divide evenly into high/low-value images ###
dset = prepare_data(all_obs, critic, device)

### Initialize networks ###
aae = AAE(X_dim=n_channels, N=n, z_dim=z_dim).to(device)
D_gauss = D_net(NUM_CLASSES, 64).to(device)

with torch.no_grad():
    aae.eval()
    class_out, _, _ = aae(torch.zeros((BATCH_SIZE, n_channels, h, h)).to(device))
    _ = D_gauss(class_out)
    aae.train()

optim_aae = torch.optim.Adam(aae.parameters(), lr=gen_lr)

#regularizing optimizers
optimizer_class = torch.optim.SGD
optim_Q_gen = optimizer_class(aae.encoder.parameters(), lr=reg_lr, momentum=0.1) # Generator
optim_D_gauss = optimizer_class(D_gauss.parameters(), lr=reg_lr, momentum=0.1) # Discriminator classification

scheduler1 = torch.optim.lr_scheduler.ConstantLR(optim_Q_gen, factor=eps, total_iters=1000)
scheduler2 = torch.optim.lr_scheduler.ConstantLR(optim_D_gauss, factor=eps, total_iters=1000)

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
        
        images = Tensor(np.array([d[0] for d in all_data])).to(device)
        labels = Tensor(np.array([d[1] for d in all_data])).to(device)
        labels = F.one_hot(labels.long(), num_classes=NUM_CLASSES)

        optim_aae.zero_grad()
        optim_Q_gen.zero_grad()
        optim_D_gauss.zero_grad()

        ### Autoencoder and Classifier ###
        class_out, z_sample, X_sample = aae(images)

        recon_loss = F.mse_loss(X_sample, images)
        c_loss = F.binary_cross_entropy(class_out, labels.float())

        total_loss = recon_loss + c_loss
        total_loss.backward()
            
        optim_aae.step()

        ### GENERATOR ###
        class_out, _ = aae.encoder(images)
        D_fake_gauss = D_gauss(class_out)
        
        # Generator loss
        G_loss = F.binary_cross_entropy_with_logits(D_fake_gauss, one_label)
        G_loss.backward()
        optim_Q_gen.step()
    
        ### DISCRIMINATOR ###
        class_out, _ = aae.encoder(images)

        D_fake_gauss = D_gauss(class_out.detach())
        z_real_gauss, _ = sample_gauss()
        D_real_gauss = D_gauss(z_real_gauss)

        # Discriminator classification loss
        D_loss_gauss = F.binary_cross_entropy_with_logits(D_real_gauss, one_label) \
                        + F.binary_cross_entropy_with_logits(D_fake_gauss, zero_label)
        
        D_loss_gauss.backward()
        optim_D_gauss.step()

        scheduler1.step()
        scheduler2.step()

        #============ TensorBoard logging ============# 
        # Log after each epoch
        # (1) Log the scalar values
        if batch_i % collect == 0:
            print(f'step {batch_i + (DATA_SAMPLES * ep)}')

            info = {
                'recon_loss': recon_loss.item(),
                'classifier_loss': c_loss.item(),
                'discriminator_loss_gauss': D_loss_gauss.item(),
                'generator_loss': G_loss.item(),
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, batch_i + (DATA_SAMPLES * ep))

        #if ep+1 == EPOCHS: # fix
            # (3) Log the images
            #info = {
            #    'images': [to_np(X_sample.view(-1, 3, h, h)[:log_count]), to_np(images_constant[:log_count])]
            #}

            #for tag, images in info.items():
            #    logger.image_summary(tag, images, ep, to_np(labels_constant))

    # Save states
    torch.save(aae.state_dict(),f'AAE_weights_{LOSS}.pt')
    #torch.save(D_gauss.state_dict(),'D_discriminator_gauss_weights.pt')