import torch
from torch import nn
from parameters import eps
import numpy as np
from PIL import Image


def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def master_params(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            yield p

# Yielded good results but not perfect.
def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            torch.nn.init.normal_(m.bias, 0, 1.414)

def prepare_rgb_image(img_array): # numpy_array
    img_array = np.transpose(img_array, (1, 2, 0)) # CHW to HWC
    img_array = (img_array * 255).astype(np.uint8)
    #Image.fromarray(img, mode='RGB').save(s, format="png")
    image = Image.fromarray(img_array, mode='RGB')

    return img_array, image

# Not fully mine. Some parts taken from another lecture.
# Tune min/max parameters if needed.
'''
def plot_latent_space(vae, wc, n=120, figsize=15, min_x=-8, min_y=-8, max_x=8, max_y=8):
    from matplotlib import pyplot as plt
    # display a n*n 2D manifold of digits
    digit_size = 32
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(min_x, max_x, n)
    grid_y = np.linspace(min_y, max_y, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            c_head = torch.Tensor(wc[torch.cdist(torch.Tensor(wc), torch.Tensor(z_sample)).argmin()])
            z_sample = torch.cat([c_head, torch.Tensor(z_sample[0])-c_head])
            x_decoded = to_np(vae(z_sample.unsqueeze(0).cuda()))
            if n_channels > 1:
              digit = x_decoded[0].reshape(3, digit_size, digit_size)
            else:
              digit = x_decoded[0].reshape(digit_size, digit_size)
            if np.any(abs(wc[:,0] - xi) < 0.125, where=(abs(wc[:,1] - yi) < 0.125)):
                digit = np.ones_like(digit)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()
'''

# Euclidean loss between cluster heads
def euclid_loss(wc):
    dists = torch.cdist(wc.float(), wc.float())
    dists = 1 / (dists + eps)
    dists[dists < 1/4] = 0 # cutoff at distance 4
    dists[dists == 1/eps] = 0 # ignore distances between same cluster-heads
    dists = torch.mean(dists)
    return dists