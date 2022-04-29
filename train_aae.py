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

### Initialize networks ###
Q = Q_net(X_dim=n_channels, N=n, z_dim=z_dim).to(device)

with torch.no_grad():
    Q.eval()
    shape = Q.get_shape(torch.zeros((BATCH_SIZE, n_channels, h, h)).to(device))
    Q.train()

P = P_net(X_dim=n_channels, N=n, z_dim=z_dim, inner_shape=shape).to(device)
#D_gauss = D_net(32, NUM_CLASSES).to(device)

with torch.no_grad():
    Q.eval()
    class_out, z_sample = Q(torch.zeros((BATCH_SIZE, n_channels, h, h)).to(device))
    _ = P(z_sample)
    #_ = D_gauss(class_out)
    Q.train()


optimizer_class = torch.optim.SGD
params = [Q.parameters(), P.parameters()]

#encode/decode optimizer
optim_P = torch.optim.Adam(P.parameters(), lr=gen_lr)
optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=gen_lr)

#regularizing optimizers
optim_Q_gen = optimizer_class(Q.parameters(), lr=reg_lr, momentum=0.1) # Generator
#optim_D_gauss = optimizer_class(D_gauss.parameters(), lr=reg_lr, momentum=0.1) # Discriminator classification

#scheduler1 = torch.optim.lr_scheduler.ConstantLR(optim_Q_gen, factor=eps, total_iters=1000)
#scheduler2 = torch.optim.lr_scheduler.ConstantLR(optim_D_gauss, factor=eps, total_iters=1000)

one_label = torch.ones((BATCH_SIZE, 1), device='cuda')
zero_label = torch.zeros((BATCH_SIZE, 1), device='cuda')

num_samples = all_obs.shape[0]
set_eval_images = True

# Start training
for ep in range(EPOCHS):
    epoch_indices = np.arange(num_samples)
    np.random.shuffle(epoch_indices)

    print(f'epoch: {ep}')

    for batch_i in range(0, num_samples, BATCH_SIZE):
        # NOTE: this will cut off incomplete batches from end of the random indices
        batch_indices = epoch_indices[batch_i:batch_i + BATCH_SIZE]

        # Load the inputs and preprocess
        images = all_obs[batch_indices].astype(np.float32)
        images = images.transpose(0, 3, 1, 2) # Transpose observations to be channel-first (BCHW instead of BHWC)
        images /= 255.0 # Normalize observations. Do this here to avoid using too much memory (images are uint8 by default)
        images = Tensor(images).to(device)

        preds = critic.evaluate(images)
        labels = get_critic_labels(preds[0])
        labels = labels.to(device)
        
        if set_eval_images:
            images_constant = images
            labels_constant = labels

            set_eval_images = False
        
        labels = F.one_hot(labels, NUM_CLASSES).float()
        
        #encode/decode optimizer
        optim_P.zero_grad()
        optim_Q_enc.zero_grad()

        #regularizing optimizers
        optim_Q_gen.zero_grad()
        #optim_D_gauss.zero_grad()

        # Autoencoder and Classifier
        class_out, z_sample = Q(images)  #encode to z
        X_sample = P(z_sample) #decode to X reconstruction
        
        recon_loss = F.mse_loss(X_sample, images)
        C_loss = F.binary_cross_entropy(class_out, labels) # Classifier loss

        total_loss = recon_loss + C_loss
        total_loss.backward()
            
        optim_P.step()
        optim_Q_enc.step()
                
        # Generator
        #class_out, z_sample = Q(images)

        #D_fake_gauss = D_gauss(class_out)   
        # Generator loss
        
        """ LEAVE OUT FOR NOW
        G_loss = F.binary_cross_entropy_with_logits(D_fake_gauss, one_label)

        G_loss.backward()
        torch.nn.utils.clip_grad_norm_(master_params(optim_Q_gen), 3.0)
        optim_Q_gen.step()
        """
        
        # Discriminators
        #D_fake_gauss = D_gauss(class_out.detach())
        #z_real_gauss = (F.one_hot((torch.rand((BATCH_SIZE), device='cuda')*NUM_CLASSES).long(), NUM_CLASSES)).float()
        #D_real_gauss = D_gauss(z_real_gauss)

        # Discriminator classification loss
        """ LEAVE OUT FOR NOW
        D_loss_gauss = F.binary_cross_entropy_with_logits(D_real_gauss, one_label) \
                        + F.binary_cross_entropy_with_logits(D_fake_gauss, zero_label)
        
        D_loss_gauss.backward()
        torch.nn.utils.clip_grad_norm_(master_params(optim_D_gauss), 3.0)
        optim_D_gauss.step()
        """
        # Remaining updates
        
        # scheduler1.step()
        # scheduler2.step()

        #============ TensorBoard logging ============# 
        # Log after each epoch
        # (1) Log the scalar values
        if batch_i % collect == 0:
            print(f'step {batch_i + (DATA_SAMPLES * ep)}')

            info = {
                'recon_loss': recon_loss.item(),
                'classifier_loss': C_loss.item(),
                #'discriminator_loss_gauss': D_loss_gauss.item(),
                #'generator_loss': G_loss.item(),
            }
        
            for tag, value in info.items():
                logger.scalar_summary(tag, value, batch_i+1)

        if ep+1 == EPOCHS:
            with torch.no_grad():
                Q.eval()
                P.eval()
                _, z_sample = Q(images_constant[:log_count])
                X_sample = P(z_sample)
                Q.train()
                P.train()

            # (3) Log the images
            info = {
                'images': [to_np(X_sample.view(-1, 3, h, h)[:log_count]), to_np(images_constant[:log_count])]
            }

            for tag, images in info.items():
                logger.image_summary(tag, images, ep, to_np(labels_constant))

    # Save states
    torch.save(Q.state_dict(),f'Q_encoder_weights_{LOSS}.pt')
    torch.save(P.state_dict(),f'P_decoder_weights_{LOSS}.pt')
    #torch.save(D_gauss.state_dict(),'D_discriminator_gauss_weights.pt')
    #torch.save(D_cat.state_dict(),'D_discriminator_cat_weights.pt')