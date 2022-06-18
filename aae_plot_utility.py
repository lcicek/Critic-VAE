import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from parameters import device, h
from utility import prepare_rgb_image, to_np

def plot_reconstructed(decoder, r0=(-5, 10), r1=(-10, 5), n=12):
    img = np.zeros((n*h, n*h, 3))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = decoder.forward(z, plot=True)
            x_hat, _ = prepare_rgb_image(to_np(x_hat.view(-1, 3, h, h)[0]))
            img[(n-1-i)*h:(n-1-i+1)*h, j*h:(j+1)*h] = x_hat
    
    plt.imshow(img.astype(np.uint8), extent=[*r0, *r1])
    plt.show()

def init_subplots():
    zero_plot = plt.subplot(121)    
    one_plot = plt.subplot(122, sharex=zero_plot, sharey=zero_plot)

    zero_plot.set_title('Critic-Label: 0')
    one_plot.set_title('Critic-Label: 1')

    return zero_plot, one_plot

def subplot(Q, subplot, images, labels, single_plot):
    with torch.no_grad():
            z = Q.get_plot_output(images).to(device)
            z = z.detach().to('cpu').numpy()

    x = z[:, 0]
    y = z[:, 1]

    if single_plot:
        subplot.scatter(x, y, c=labels)
    else:
        # Calculate the point density
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        subplot.scatter(x, y, c=z)

def separate(images, labels):
    zero_indices = torch.from_numpy(np.where(labels==0)[0]).to(device).long()
    one_indices = torch.from_numpy(np.where(labels==1)[0]).to(device).long()

    zero_labels = torch.zeros((len(zero_indices), 1))
    one_labels = torch.ones((len(one_indices), 1))

    zero_images = torch.index_select(input=images, dim=0, index=zero_indices)
    one_images = torch.index_select(input=images, dim=0, index=one_indices)

    return zero_images, zero_labels, one_images, one_labels


