import torch
from torch import nn, Tensor
from parameters import BATCH_SIZE, DATA_SAMPLES, CRIT_THRESHOLD, MINERL_SAMPLES, LHV_IMG_COUNT
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

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

def sample_gauss():
    sample = torch.rand((BATCH_SIZE, 2), device='cuda')
    labels = torch.clone(sample)
    
    labels[labels < CRIT_THRESHOLD] = 0 # already one hot version
    labels[labels >= CRIT_THRESHOLD] = 1

    return sample, labels


def get_critic_labels(preds):
    labels = []
    for pred_value in preds:
        if pred_value >= CRIT_THRESHOLD:
            label = 1
        else:
            label = 0

        labels.append(label)

    return torch.as_tensor(labels)

def prepare_data(data, critic, device):
    print('preparing data...')
    final_dset = []
    num_samples = data.shape[0]

    low_value_images = []
    high_value_images = []

    for batch_i in range(0, num_samples, BATCH_SIZE):        
        # Preprocess
        images = data[batch_i:batch_i+BATCH_SIZE].astype(np.float32)
        images = images.transpose(0, 3, 1, 2) # Transpose observations to be channel-first (BCHW instead of BHWC)
        images /= 255.0 # Normalize observations. Do this here to avoid using too much memory (images are uint8 by default)
        images = Tensor(images).to(device)

        preds, _ = critic.evaluate(images)
        labels = get_critic_labels(preds)
        images = images.detach().cpu().numpy()

        # Save (img, label)-tuple for low/high-value images respectively
        low_value_images.extend((images[i], label) for i, label in enumerate(labels) if label == 0)
        high_value_images.extend((images[i], label) for i, label in enumerate(labels) if label == 1)

    # Make sure enough images were collected
    assert len(low_value_images) >= LHV_IMG_COUNT
    assert len(high_value_images) >= LHV_IMG_COUNT

    # Randomize which images get chosen
    np.random.shuffle(low_value_images)
    np.random.shuffle(high_value_images)

    final_dset.extend(low_value_images[0:LHV_IMG_COUNT])
    final_dset.extend(high_value_images[0:LHV_IMG_COUNT])
    final_dset = np.array(final_dset, dtype=object)
    
    # Randomize order of high and low value images
    np.random.shuffle(final_dset)

    return final_dset

# source: https://github.com/KarolisRam/MineRL2021-Research-baselines/blob/main/standalone/Behavioural_cloning.py#L105
def load_minerl_data(data):
    print("loading minerl-data...")
    trajectory_names = data.get_trajectory_names()
    random.shuffle(trajectory_names)

    all_pov_obs = []
    # Add trajectories to the data until we reach the required DATA_SAMPLES.
    for trajectory_name in trajectory_names:
        trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
        for dataset_observation, _, _, _, _ in trajectory:
            all_pov_obs.append(dataset_observation["pov"])
        if len(all_pov_obs) >= MINERL_SAMPLES:
            length = len(all_pov_obs)
            excess = length % BATCH_SIZE
            all_pov_obs = all_pov_obs[:length-excess] # cut off excess
            break

    all_pov_obs = np.array(all_pov_obs)

    return all_pov_obs

def plot_latent(autoencoder, data, num_batches, device):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break

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