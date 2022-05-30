from operator import index
import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde
import sys

from nets import *
from parameters import *
from utility import *

SEPARATE_IMAGES = True
NORMALIZE = False
DECODER = True

def plot_reconstructed(decoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 16
    img = np.zeros((n*w, n*w, 3))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = decoder.forward(z, plot=True)
            x_hat, _ = prepare_rgb_image(to_np(x_hat.view(-1, 3, w, w)[0]))
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    
    plt.imshow(img.astype(np.uint8), extent=[*r0, *r1])
    plt.show()

def init_subplots():
    zero_plot = plt.subplot(121)    
    one_plot = plt.subplot(122, sharex=zero_plot, sharey=zero_plot)

    zero_plot.set_title('Critic-Label: 0')
    one_plot.set_title('Critic-Label: 1')

    #zero_plot.axis(xmin=-2, xmax=2)

    return zero_plot, one_plot

def subplot(subplot, images, labels):
    with torch.no_grad():
            z = Q.get_plot_output(images).to(device) # 32 dimensional
            z = z.detach().to('cpu').numpy()

            #if NORMALIZE:
            #    # Normalize data to range [0, 1] so plotting doesn't vary
            #    z_min = np.min(z, axis=0, keepdims=True) # axis 0 since shape is batch_size, _, _, _
            #    z_max_min = np.max(z, axis=0, keepdims=True) - z_min
            #    z = (z - z_min) / z_max_min

    x = z[:, 0]
    y = z[:, 1]

    if SEPARATE_IMAGES:
        # Calculate the point density
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        subplot.scatter(x, y, c=z)
    else:
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

if DECODER:
    try:
        P = P_net(X_dim=n_channels, N=n, z_dim=z_dim).to(device)
        P.load_state_dict(torch.load('P_decoder_weights.pt'))
    except Exception as e:
        print(e)

    plot_reconstructed(P)
    sys.exit()

critic = Critic().to(device)
critic.load_state_dict(torch.load(CRITIC_PATH, map_location=device))

Q = Q_net(X_dim=n_channels, N=n, z_dim=z_dim).to(device)

try:
    dset = torch.load('plot_dataset.pt')
    Q.load_state_dict(torch.load('Q_encoder_weights.pt'))
except Exception as e:
    print('error occured')
    print(e) 

Q.eval()

if SEPARATE_IMAGES:
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

        if SEPARATE_IMAGES:
            zi, zl, oi, ol = separate(images=images, labels=labels)
            subplot(zero_plot, zi, zl)
            subplot(one_plot, oi, ol)
        else:
            subplot(plt, images, labels)

        if batch_i >= num_samples - BATCH_SIZE:
            #if NORMALIZE:
            #plt.xlim([-2, 2]) # x-coordinate boundaries
            #plt.ylim([-2, 2])

            if not SEPARATE_IMAGES:
                ax = plt.gca() #get the current axes
                pcm = ax.get_children()[2]
                plt.colorbar(pcm, ax=ax)

            plt.show()
            break