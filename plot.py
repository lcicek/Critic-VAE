import torch
from torch import Tensor
import os
import minerl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from nets import *
from parameters import *
from utility import *


try:
    matplotlib.use('MacOSX')
    print('using macosx backend')
except:
    print('failed to use macosx backend')
    print(f'using {matplotlib.get_backend()} instead')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Initialize mineRL dataset ###
os.environ['MINERL_DATA_ROOT'] = MINERL_DATA_ROOT_PATH
data = minerl.data.make('MineRLTreechop-v0', num_workers=1)
all_obs = load_minerl_data(data)
del data

critic = Critic().to(device)
critic.load_state_dict(torch.load(CRITIC_PATH, map_location=device))
dset = prepare_data(all_obs, critic, device)

aae = AAE(X_dim=n_channels, N=n, z_dim=z_dim).to(device)

try:
    aae.load_state_dict(torch.load('AAE_weights_RCD.pt'))
except Exception as e:
    print(e) 

aae.eval()

print('plotting...')
num_samples = dset.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)
for batch_i in range(0, num_samples, BATCH_SIZE):
        # NOTE: this will cut off incomplete batches from end of the random indices
        batch_indices = indices[batch_i:batch_i + BATCH_SIZE]
        all_data = dset[batch_indices]

        images = Tensor(np.array([d[0] for d in all_data])).to(device)
        labels = np.array([d[1] for d in all_data])

        z = aae.encoder.get_plot_output(images)
        z = z.detach().to('cpu').numpy()

        x = z[:, 0]
        y = z[:, 1]

        plt.scatter(x, y, c=labels)
        if batch_i >= num_samples - BATCH_SIZE:
            plt.colorbar()
            plt.show()
            break