import itertools
import torch
import torch.nn.functional as F
from torch.cuda.amp.grad_scaler import GradScaler
from torch import Tensor
import time
import os
import minerl
from minerl.data import BufferedBatchIter
import numpy as np
import random
#import cv2
#import tensorflow as tf

from nets import Q_net, P_net, D_net, Critic
from logger import Logger
from parameters import (
    CRITIC_PATH, MINERL_DATA_ROOT_PATH,
    NUM_CLASSES, BATCH_SIZE, TRAIN, AUTOCAST, EPOCHS, 
    n_channels, eps, n, z_dim, total_step, collect, log_count
)
from utility import (
    to_np, to_var, euclid_loss, master_params, init_weights, 
    load_minerl_data, get_critic_labels
) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# minerl dataset
os.environ['MINERL_DATA_ROOT'] = MINERL_DATA_ROOT_PATH
data = minerl.data.make('MineRLTreechop-v0')
all_obs = load_minerl_data(data) # get minecraft observations

# load critic
critic = Critic().to(device)
critic.load_state_dict(torch.load(CRITIC_PATH, map_location=device))

Q = Q_net(X_dim=n_channels, N=n, z_dim=z_dim).to(device)

with torch.no_grad():
    Q.eval()
    shape, combined_shape = Q.get_shape(torch.zeros((BATCH_SIZE, n_channels, 32, 32)).to(device))
    Q.train()

P = P_net(X_dim=n_channels, N=n, z_dim=z_dim, inner_shape=shape).to(device)
D_gauss = D_net(32, NUM_CLASSES).to(device)

with torch.no_grad():
    Q.eval()
    class_out, z_sample = Q.custom_forward(torch.zeros((BATCH_SIZE, n_channels, 32, 32)).to(device))
    _ = P(z_sample)
    _ = D_gauss(class_out)
    Q.train()
    init_weights(Q)
    init_weights(P)
    init_weights(D_gauss)

if TRAIN:
    # Set learning rates
    gen_lr = 0.001
    reg_lr = 0.001

    optimizer_class = torch.optim.SGD
        
    params = [Q.parameters(), P.parameters()]
    
    #encode/decode optimizer
    optim_QP = torch.optim.Adam(itertools.chain(*params), lr=gen_lr)
    
    #regularizing optimizers
    optim_Q_gen = optimizer_class(Q.parameters(), lr=reg_lr, momentum=0.1) # Generator
    optim_D_gauss = optimizer_class(D_gauss.parameters(), lr=reg_lr, momentum=0.1) # Discriminator classification

    scheduler1 = torch.optim.lr_scheduler.ConstantLR(optim_Q_gen, factor=eps, total_iters=1000)
    scheduler2 = torch.optim.lr_scheduler.ConstantLR(optim_D_gauss, factor=eps, total_iters=1000)
    
    if AUTOCAST:
        scaler_Q_enc = GradScaler()
        scaler_Q_gen = GradScaler()
        scaler_D_gauss = GradScaler()
        scaler_D_cat = GradScaler()

    logger = None

if TRAIN:
    one_label = torch.ones((BATCH_SIZE, 1), device='cuda')
    zero_label = torch.zeros((BATCH_SIZE, 1), device='cuda')

    num_samples = all_obs.shape[0]
    set_eval_images = True

    # Start training
    for _ in range(EPOCHS):
        epoch_indices = np.arange(num_samples)
        np.random.shuffle(epoch_indices)

        for batch_i in range(0, num_samples, BATCH_SIZE):
            # NOTE: this will cut off incomplete batches from end of the random indices
            batch_indices = epoch_indices[batch_i:batch_i + BATCH_SIZE]

            # Load the inputs and preprocess
            images = all_obs[batch_indices].astype(np.float32)
            images = images.transpose(0, 3, 1, 2) # Transpose observations to be channel-first (BCHW instead of BHWC)
            images /= 255.0 # Normalize observations. Do this here to avoid using too much memory (images are uint8 by default)
            images = Tensor(images).to(device)

            preds = critic.evaluate(images)
            labels = get_critic_labels(preds[0]).to(device)
            
            if set_eval_images:
                images_constant = Tensor(all_obs[batch_indices]).to(device)
                labels_constant = labels

                set_eval_images = False
            
            # labels = F.one_hot(labels, NUM_CLASSES).float()
            
            #encode/decode optimizer
            optim_QP.zero_grad()
            
            #regularizing optimizers
            optim_Q_gen.zero_grad()
            optim_D_gauss.zero_grad()

            # Autoencoder and Classifier; might need cuda here? (!)
            #with torch.autocast('cuda', AUTOCAST):
            class_out, z_sample = Q.custom_forward(images)  #encode to z
            X_sample = P(z_sample) #decode to X reconstruction
            
            recon_loss = F.mse_loss(X_sample, images) * 10

            # Euclid loss
            E_loss = euclid_loss(Q.wc) * 5
            
            # Classifier loss
            C_loss = F.cross_entropy(class_out, labels)
            total_loss = recon_loss + C_loss + E_loss
            
            if AUTOCAST:
                scaler_Q_enc.scale(total_loss).backward()
                scaler_Q_enc.step(optim_QP)
            else:
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(master_params(optim_QP), 3.0)
                optim_QP.step()
                    
            # Generator
            #with torch.autocast('cuda', AUTOCAST):
            class_out, z_sample = Q.custom_forward(images)
            
            D_fake_gauss = D_gauss(class_out)

            # Generator loss
            G_loss = F.binary_cross_entropy_with_logits(D_fake_gauss, one_label)

            if AUTOCAST:
                scaler_Q_gen.scale(G_loss).backward()
                scaler_Q_gen.step(optim_Q_gen)

            else:
                G_loss.backward()
                torch.nn.utils.clip_grad_norm_(master_params(optim_Q_gen), 3.0)
                optim_Q_gen.step()

            
            # Discriminators
            #with torch.autocast('cuda', AUTOCAST):
            D_fake_gauss = D_gauss(class_out.detach())
            z_real_gauss = (F.one_hot((torch.rand((BATCH_SIZE), device='cuda')*NUM_CLASSES).long(), NUM_CLASSES)).float()
            D_real_gauss = D_gauss(z_real_gauss)

            # Discriminator classification loss
            D_loss_gauss = F.binary_cross_entropy_with_logits(D_real_gauss, one_label) \
                            + F.binary_cross_entropy_with_logits(D_fake_gauss, zero_label)
            
            if AUTOCAST:
                scaler_D_gauss.scale(D_loss_gauss).backward()
                scaler_D_gauss.step(optim_D_gauss)
            else:
                D_loss_gauss.backward()
                torch.nn.utils.clip_grad_norm_(master_params(optim_D_gauss), 3.0)
                optim_D_gauss.step()
            
            # Remaining updates
            if AUTOCAST:
                scaler_Q_enc.update()
                scaler_Q_gen.update()   
                scaler_D_cat.update()
                scaler_D_gauss.update()
            
            scheduler1.step()
            scheduler2.step()

                #============ TensorBoard logging ============#
            if (batch_i+1) % collect == 0:   
                # (1) Log the scalar values
                if logger == None or batch_i+1 == collect:
                    from time import time
                    # Set the logger
                    logger = Logger('./logs/aae_with_cat' + str(time())[-5::])
                    
                
                info = {
                    'recon_loss': recon_loss.item(),
                    'discriminator_loss_gauss': D_loss_gauss.item(),
                    'generator_loss': G_loss.item(),
                    'classifier_loss': C_loss.item(),
                    'euclidean_loss': E_loss.item(),
                }
                info_arr = list(info.values())
                print('Step [%d/%d]; Losses: Recon: %.2e, D_Class: %.2e, Generator: %.2e, Classifier: %.2e, Euclid: %.2e'
                        %(batch_i+1, total_step, info_arr[0], info_arr[1], info_arr[2], 
                        info_arr[3], info_arr[4]))
                
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, batch_i+1)

                # Disabled for now, probably safe to reenable if needed
                # (2) Log values and gradients of the parameters (histogram)
                #for net,name in zip([P,Q,D_gauss],['P_','Q_','D_']): 
                #    for tag, value in net.named_parameters():
                #        tag = name+tag.replace('.', '/')
                #        logger.histo_summary(tag, to_np(value), step+1)
                #        logger.histo_summary(tag+'/grad', to_np(value.grad), step+1)
                
                #with torch.autocast('cuda', AUTOCAST):
                with torch.no_grad():
                    Q.eval()
                    P.eval()
                    _, z_sample = Q.custom_forward(images_constant[:log_count])
                    X_sample = P(z_sample)
                    Q.train()
                    P.train()

                # (3) Log the images
                ## Remove '3' if using MNIST
                info = {
                    'images': [to_np(X_sample.view(-1, 3, 32, 32)[:log_count]), to_np(images_constant[:log_count])]
                }

                for tag, images in info.items():
                    logger.image_summary(tag, images, batch_i+1, to_np(labels_constant))

                # Save states
                #torch.save(Q.state_dict(),'Q_encoder_weights.pt')
                #torch.save(P.state_dict(),'P_decoder_weights.pt')
                #torch.save(D_gauss.state_dict(),'D_discriminator_gauss_weights.pt')
                #torch.save(D_cat.state_dict(),'D_discriminator_cat_weights.pt')

else:
    print("No training to not override saved weights!")
    print("If you want to train, please set TRAIN = True.")