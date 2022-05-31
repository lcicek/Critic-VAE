import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse

from nets import *
from parameters import *
from utility import *
from plot_utility import *

parser = argparse.ArgumentParser()
parser.add_argument('-d', action='store_true') # decoder
parser.add_argument('--single', action='store_true')
args = parser.parse_args()

SINGLE_PLOT = args.single
DECODER = args.d

try:
    matplotlib.use('TkAgg') # stops working randomly after a while, works again after relogging into PC
except:
    print(f'matplotlib is using {matplotlib.get_backend()}.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DECODER:
    try:
        P = P_net(X_dim=n_channels, N=n, z_dim=z_dim).to(device)
        P.load_state_dict(torch.load(P_PATH))
    except Exception as e:
        print(e)

    plot_reconstructed(P)
else:
    critic = Critic().to(device)
    critic.load_state_dict(torch.load(CRITIC_PATH, map_location=device))

    Q = Q_net(X_dim=n_channels, N=n, z_dim=z_dim).to(device)

    try:
        dset = torch.load(PLOT_DATASET_PATH)
        Q.load_state_dict(torch.load(Q_PATH))
    except Exception as e:
        print(e) 

    Q.eval()

    print('plotting...')

    if not SINGLE_PLOT:
        zero_plot, one_plot = init_subplots()

    num_samples = dset.shape[0]
    indices = np.arange(num_samples)
    #np.random.shuffle(indices)
    for batch_i in range(0, num_samples, BATCH_SIZE):
            # NOTE: this will cut off incomplete batches from end of the random indices
            batch_indices = indices[batch_i:batch_i + BATCH_SIZE]
            batch = dset[batch_indices]

            images = Tensor(np.array([d[0] for d in batch])).to(device)
            labels = np.array([d[1] for d in batch])

            if SINGLE_PLOT:
                subplot(Q, plt, images, labels, SINGLE_PLOT)
            else:
                zi, zl, oi, ol = separate(images=images, labels=labels)
                subplot(Q, zero_plot, zi, zl, SINGLE_PLOT)
                subplot(Q, one_plot, oi, ol, SINGLE_PLOT)

            if batch_i >= num_samples - BATCH_SIZE:

                if SINGLE_PLOT:
                    ax = plt.gca() #get the current axes
                    pcm = ax.get_children()[2]
                    plt.colorbar(pcm, ax=ax)

                plt.show()
                break