import itertools
import torch
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.cuda.amp.grad_scaler import GradScaler
import time
#import cv2
#import tensorflow as tf

from utility import to_np, to_var, euclid_loss, master_params, init_weights
from nets import Q_net, P_net, D_net
from logger import Logger
from parameters import (
    NUM_CLASSES, BATCH_SIZE, TRAIN, AUTOCAST,
    n_channels, eps, n, z_dim, total_step, collect, log_count
)

# Dataset 
dataset = dsets.SVHN(root='./data', 
                      transform=transforms.Compose([
                        transforms.ToTensor(),
                        #transforms.Pad(2)  # For MNIST!
                      ]), split="train",
                      download=True)
                    
# Data Loader (Input Pipeline)
data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                          batch_size=BATCH_SIZE,
                                          num_workers=1,
                                          shuffle=True,
                                          drop_last=True)

Q = Q_net(X_dim=n_channels, N=n, z_dim=z_dim).cuda()

with torch.no_grad():
    Q.eval()
    shape, combined_shape = Q.get_shape(torch.zeros((BATCH_SIZE, n_channels, 32, 32)).cuda())
    Q.train()

P = P_net(X_dim=n_channels, N=n, z_dim=z_dim, inner_shape=shape).cuda()
D_gauss = D_net(32, NUM_CLASSES).cuda()

with torch.no_grad():
    Q.eval()
    class_out, z_sample = Q.custom_forward(torch.zeros((BATCH_SIZE, n_channels, 32, 32)).cuda())
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

    data_iter = iter(data_loader)
    iter_per_epoch = len(data_loader)

    logger = None
    
    images_constant = next(data_iter)
    images_constant, labels_constant = to_var(images_constant[0]), to_var(images_constant[1])


if TRAIN:
    one_label = torch.ones((BATCH_SIZE, 1), device='cuda')
    zero_label = torch.zeros((BATCH_SIZE, 1), device='cuda')

    # Start training
    step = 0
    while step < total_step:
      for i, data in enumerate(data_loader):
        step += 1

        # Fetch the images and labels and convert them to variables
        images, labels = data
        
        images, labels = to_var(images), to_var(labels) # labels beween 0-9
        
        
        labels = F.one_hot(labels, NUM_CLASSES).float()
        
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
        if (step+1) % collect == 0:   
            # (1) Log the scalar values
            if logger == None or step+1 == collect:
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
                  %(step+1, total_step, info_arr[0], info_arr[1], info_arr[2], 
                  info_arr[3], info_arr[4]))
            
            for tag, value in info.items():
                logger.scalar_summary(tag, value, step+1)

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
                logger.image_summary(tag, images, step+1, to_np(labels_constant))

            # Save states
            #torch.save(Q.state_dict(),'Q_encoder_weights.pt')
            #torch.save(P.state_dict(),'P_decoder_weights.pt')
            #torch.save(D_gauss.state_dict(),'D_discriminator_gauss_weights.pt')
            #torch.save(D_cat.state_dict(),'D_discriminator_cat_weights.pt')

else:
    print("No training to not override saved weights!")
    print("If you want to train, please set TRAIN = True.")