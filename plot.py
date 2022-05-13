from operator import index
import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from nets import *
from parameters import *
from utility import *

SEPERATE_IMAGES = True
NORMALIZE = False

def init_subplots():
    zero_plot = plt.subplot(121)
    one_plot = plt.subplot(122)

    zero_plot.set_title('Critic-Label: 0')
    one_plot.set_title('Critic-Label: 1')

    zero_plot.set_aspect('equal')
    one_plot.set_aspect('equal')

    zero_plot.axis(xmin=-2, xmax=2)
    zero_plot.axis(ymin=-2, ymax=2)

    one_plot.axis(xmin=-2, xmax=2)
    one_plot.axis(ymin=-2, ymax=2)

    return zero_plot, one_plot

def subplot(subplot, images, labels):
    with torch.no_grad():
            z = Q.get_plot_output(images)
            z = z.detach().to('cpu').numpy()

            if NORMALIZE:
                # Normalize data to range [0, 1] so plotting doesn't vary
                z_min = np.min(z, axis=0, keepdims=True) # axis 0 since shape is batch_size, _, _, _
                z_max_min = np.max(z, axis=0, keepdims=True) - z_min
                z = (z - z_min) / z_max_min

    x = z[:, 0]
    y = z[:, 1]

    subplot.scatter(x, y, c=labels)

def separate(images, labels):
    zero_indices = torch.from_numpy(np.where(labels==0)[0]).to(device).long()
    one_indices = torch.from_numpy(np.where(labels==1)[0]).to(device).long()

    zero_labels = torch.zeros((len(zero_indices), 1))
    one_labels = torch.ones((len(one_indices), 1))

    zero_images = torch.index_select(input=images, dim=0, index=zero_indices)
    one_images = torch.index_select(input=images, dim=0, index=one_indices)

    return zero_images, zero_labels, one_images, one_labels

try:
    matplotlib.use('TkAgg') # stops working randomly after a while, works again after relogging into PC
    print('using tkagg backend')
except:
    print(f'using {matplotlib.get_backend()} instead')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

critic = Critic().to(device)
critic.load_state_dict(torch.load(CRITIC_PATH, map_location=device))

Q = Q_net(X_dim=n_channels, N=n, z_dim=z_dim).to(device)

try:
    dset = torch.load('plot_dataset.pt')
    Q.load_state_dict(torch.load('Q_encoder_weights_RCD.pt'))
except Exception as e:
    print(e) 

Q.eval()

if SEPERATE_IMAGES:
    zero_plot, one_plot = init_subplots()

print('plotting...')
num_samples = dset.shape[0]
indices = np.arange(num_samples)
#np.random.shuffle(indices)
for batch_i in range(0, num_samples, BATCH_SIZE):
        # NOTE: this will cut off incomplete batches from end of the random indices
        batch_indices = indices[batch_i:batch_i + BATCH_SIZE]
        batch = dset[batch_indices]

        images = Tensor(np.array([d[0] for d in batch])).to(device)
        labels = np.array([d[1] for d in batch])

        if SEPERATE_IMAGES:
            zi, zl, oi, ol = separate(images=images, labels=labels)
            subplot(zero_plot, zi, zl)
            subplot(one_plot, oi, ol)
        else:
            subplot(plt, images, labels)

        if batch_i >= num_samples - BATCH_SIZE:
            #if NORMALIZE:
            #plt.xlim([-2, 2]) # x-coordinate boundaries
            #plt.ylim([-2, 2])

            if not SEPERATE_IMAGES:
                ax = plt.gca() #get the current axes
                pcm = ax.get_children()[2]
                plt.colorbar(pcm, ax=ax)

            plt.show()
            break







