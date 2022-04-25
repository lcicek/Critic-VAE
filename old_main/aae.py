import torch
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import Tensor
from logger import Logger
import itertools
import os
import sys
from PIL import Image
import numpy as np

import minerl
from minerl.data import BufferedBatchIter

from nets import Q_net, P_net, D_net, NewCritic
from utility import to_np, pred_to_label, init_weights, euclid_loss, master_params, gauss_sample

critic_path = '/homes/lcicek/Desktop/AAE/critic-rewidx=1-cepochs=15-datamode=trunk-datasize=99999-shift=12-chfak=1-dropout=0.3.pt'
os.environ['MINERL_DATA_ROOT'] = '/homes/lcicek/anaconda3/envs/aae/lib/python3.9/site-packages/minerl'
img_path = '/homes/lcicek/Desktop/AAE/test_images'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 2
EPS = 1e-7 # 1e-7?
z_dim = 256 # 120?
n_channels = 3 # input dimension (3 because of RGB channels)
n = 24

### INITIALIZE NETWORKS ###

Q = Q_net(n_channels, n, z_dim).to(device)

with torch.no_grad():
    Q.eval()
    shape, _ = Q.get_shape(torch.zeros((1, n_channels, 64, 64)).to(device))
    Q.train()

P = P_net(n_channels, n, z_dim, shape).to(device)
# D = D_net(32, 2).to(device) # 2 classes

with torch.no_grad():
    Q.eval()
    init, model = Q(torch.zeros((1, n_channels, 64, 64)).to(device))
    _ = P(model)
    #_ = D(init)
    Q.train()

    init_weights(Q)
    init_weights(P)
    #init_weights(D)

# Set the logger
logger = Logger('./logs/z_120_fixed_LR_2')

# Set learning rates
gen_lr = 0.001
reg_lr = 0.001

### OPTIMIZERS ###
optimizer_class = torch.optim.SGD
params = [Q.parameters(), P.parameters()]

#encode/decode optimizer
optim_QP = torch.optim.Adam(itertools.chain(*params), lr=gen_lr)

#regularizing optimizers
optim_Q_gen = optimizer_class(Q.parameters(), lr=reg_lr, momentum=0.1) # gen = Generator
#optim_D = optimizer_class(D.parameters(), lr=reg_lr, momentum=0.1)

scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optim_Q_gen, milestones=[2, 5], gamma=0.1)
#scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optim_D, milestones=[2,5], gamma=0.1)

### SETUP CRITIC NETWORK ### 
critic = NewCritic().to(device)
critic.load_state_dict(torch.load(critic_path, map_location=device))

data = minerl.data.make('MineRLTreechop-v0')
iterator = BufferedBatchIter(data)
all_trajectories = iterator.all_trajectories.copy() # 209 trajectories in total

epochs = 1
total_step = 100 # steps per trajectory

step = 0
total_images = 0
traj_count = 0

### TRAINING ###
for _ in range(epochs):
    for trajectory in all_trajectories:
        if traj_count >= 1:
            break
        traj_count += 1

        step = 0
        iterator = BufferedBatchIter(data) # reset every iteration
        iterator.available_trajectories = [trajectory]

        ### RETRIEVE IMAGES FROM TRAJECTORY AND DO TRAINING ###
        bb_iter = iterator.buffered_batch_iter(batch_size=1, num_epochs=1) # idas can only handle batch size 1

        try:
            for current_state, _, _, _, _ in bb_iter:
                if step >= total_step: # finished
                    break
                step += 1
                total_images += 1

                optim_QP.zero_grad()
                optim_Q_gen.zero_grad()
                #optim_D.zero_grad()

                P.zero_grad()
                Q.zero_grad()
                #D.zero_grad()

                images = Tensor(current_state['pov']).to(device) # batch_size amount of images
                preds = critic.evaluate(images) # preds[0]=preds of all images, preds[1]= 5 layers with all sub-preds of the layers
                label = pred_to_label(preds[0])

                images = images.permute(0, 3, 1, 2)

                ### LOSSES ### 
                z_class, z_rep = Q(images) # encode to z
                X_sample = P(z_rep) # decode to X reconstruction

                recon_loss = 10 * F.mse_loss(X_sample, images) # reconstruction loss
                # e_loss = 5 * euclid_loss(Q.wc) # euclid loss
                c_loss = F.cross_entropy(z_class, label.long()) # classifier loss

                total_loss = recon_loss + c_loss # + e_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(master_params(optim_QP), 3.0)
                optim_QP.step()

                ### GENERATOR LOSS ###
                #z_class, z_rep = Q(reshaped) # necessary to call again?
                #D_fake = D(z_class) # shape: [1, 1]

                #G_loss = F.binary_cross_entropy_with_logits(D_fake, label.view(1, 1)) # generator loss
                #G_loss.backward()
                #torch.nn.utils.clip_grad_norm_(master_params(optim_Q_gen), 3.0)
                #optim_Q_gen.step()

                ### DISCRIMINATOR LOSS ###
                #D_fake = D(z_class.detach())
                #D_loss = F.binary_cross_entropy_with_logits(D_fake, gauss_sample())
                #D_loss.backward()
                #torch.nn.utils.clip_grad_norm_(master_params(optim_D), 3.0)
                #optim_D.step()

                scheduler1.step()
                #scheduler2.step()
                
                if total_images % 100 == 0:
                    print(f'images: {total_images}')

                    #============ TensorBoard logging ============#
                    # (1) Log the scalar values
                    info = {
                        'recon_loss': recon_loss.data.item(),
                        'classifier_loss': c_loss.data.item(),
                        # 'euclidean_loss': e_loss.data.item(),
                        'total_loss': total_loss.data.item(),
                        #'discriminator_loss': D_loss.data.item(),
                        #'generator_loss': G_loss.data.item()
                    }

                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, total_images)

                    with torch.no_grad():
                        Q.eval()
                        P.eval()
                        z_class, z_rep = Q(images) # ?
                        X_sample = P(z_rep)
                        Q.train()
                        P.train()

                    # (3) Log the images
                    info = {
                        'imgs': to_np(X_sample.view(1, 64, 64, 3)) # [:10]
                    }

                    # save image
                    #test_images = to_np(X_sample.view(1, 64, 64, 3))
                    #saved_img_arr = test_images[0]
                    #pil_img = Image.fromarray(np.uint8(saved_img_arr))
                    #pil_img.save(f'{img_path}/{total_images}.png')

                    for tag, imgs in info.items():
                        logger.image_summary(tag, imgs, total_images)

                    # (2) Log values and gradients of the parameters (histogram)
                    #for net,name in zip([P,Q,D],['P_','Q_','D_']): 
                    #    for tag, value in net.named_parameters():
                    #        tag = name+tag.replace('.', '/')
                    #        logger.histo_summary(tag, to_np(value), traj_count)
                    #        logger.histo_summary(tag+'/grad', to_np(value.grad), traj_count)
        except RuntimeError as e:
            print(e)
            print('continuing...')

torch.save(Q.state_dict(),'Q_encoder_weights.pt')
torch.save(P.state_dict(),'P_decoder_weights.pt')
#torch.save(D.state_dict(),'D_discriminator_weights.pt')
