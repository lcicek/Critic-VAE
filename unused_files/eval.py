import torch
from torch import Tensor
import numpy as np

from aae_nets import *
from parameters import *
from utility import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

critic = Critic().to(device)
critic.load_state_dict(torch.load(CRITIC_PATH, map_location=device))

Q = Q_net(X_dim=n_channels, N=n, z_dim=z_dim).to(device)
P = P_net(X_dim=n_channels, N=n, z_dim=z_dim).to(device)
D = D_net(32, 32).to(device)

try:
    dset = torch.load('plot_dataset.pt')
    Q.load_state_dict(torch.load('saved-networks/Q_encoder_weights.pt'))
    P.load_state_dict(torch.load('saved-networks/P_decoder_weights.pt'))
    D.load_state_dict(torch.load('saved-networks/D_discriminator_weights.pt'))
except Exception as e:
    print(e) 

Q.eval()
P.eval()
D.eval()

num_samples = dset.shape[0]
indices = np.arange(num_samples)
#np.random.shuffle(indices)
for batch_i in range(0, num_samples, BATCH_SIZE):
        # NOTE: this will cut off incomplete batches from end of the random indices
        batch_indices = indices[batch_i:batch_i + BATCH_SIZE]
        batch = dset[batch_indices]

        images = Tensor(np.array([d[0] for d in batch])).to(device)
        labels = np.array([d[1] for d in batch])

        _, z_sample = Q(images)
        X_sample = P(z_sample)

        for i, img in enumerate(X_sample):
            conc_h = np.concatenate((to_np(images[i].view(-1, 3, h, h)[0]), to_np(X_sample[i].view(-1, 3, h, h)[0])), axis=2)
            _, img = prepare_rgb_image(conc_h)

            img.save(f'{SAVE_PATH}/{labels[i]}-img{i:03d}.png', format="png")