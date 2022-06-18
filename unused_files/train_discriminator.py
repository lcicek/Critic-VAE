import torch
import torch.nn.functional as F
from torch import Tensor
import time
import os
import minerl
import numpy as np
from time import time

from logger import Logger
from aae_nets import *
from parameters import *
from utility import *

LEARNING_RATE = 0.001

logger = Logger('./logs/disc' + str(time())[-5::])

### Initialize mineRL dataset ###
os.environ['MINERL_DATA_ROOT'] = MINERL_DATA_ROOT_PATH
data = minerl.data.make('MineRLTreechop-v0', num_workers=1)
all_pov_obs = load_minerl_data(data) # get all minerl observations
del data # without this line errors gets thrown at the end of the program

### Load trained critic model ###
print('loading critic...')
critic = Critic().to(torch.device('cpu'))
critic.load_state_dict(torch.load(CRITIC_PATH, map_location='cpu'))

### Preprocess minerl data; Divide evenly into high/low-value images ###
dset = prepare_data(all_pov_obs, critic)

### Initialize networks ###
D = D_net(32, 32).to(device)

#regularizing optimizers
optimizer_class = torch.optim.SGD
optim_D = optimizer_class(D.parameters(), lr=LEARNING_RATE, momentum=0.25) # Discriminator classification

one_label = torch.ones((BATCH_SIZE, 1), device='cuda')

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

        ### DISCRIMINATOR ###
        optim_D.zero_grad()

        z_real = sample_gauss(labels)
        D_real = D(z_real)

        loss =  F.binary_cross_entropy(D_real, one_label)
        loss.backward()

        optim_D.step()

        #============ TensorBoard logging ============# 
        # Log after each epoch
        # (1) Log the scalar values
        if batch_i % collect == 0:
            print(f'step {batch_i + (DATA_SAMPLES * ep)}')

            info = {
                'discriminator_loss': loss.item(),
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, batch_i + (DATA_SAMPLES * ep))

    # Save states
    torch.save(D.state_dict(),'D_discriminator_weights.pt')