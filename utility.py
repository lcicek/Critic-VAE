import torch
from torch import nn, Tensor
from parameters import *
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import cv2

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

def prior_sample(label):
    sample = torch.rand(z_dim).to(device)

    if label == 0:
        sample = (sample * -0.5) + 0.5 # to range [0, 0.5]

    else:
        sample = (sample * -0.5) + 1.5 # to range [1.0, 1.5]

    return sample

def sample_gauss(labels):
    # return torch.randn(BATCH_SIZE, 32, 1, 1).to(device)

    all_samples = torch.empty(BATCH_SIZE, z_dim, 1, 1).to(device) # bottleneck shape
    unflatten = nn.Unflatten(0, bottleneck)

    for i, label in enumerate(labels): # loop is very slow
        sample = prior_sample(label)
        sample = unflatten(sample)
        all_samples[i] = sample

    return all_samples

def get_critic_labels(preds):
    labels = []
    for pred_value in preds:
        if pred_value >= CRIT_THRESHOLD:
            label = 1
        else:
            label = 0

        labels.append(label)

    return torch.as_tensor(labels)

def prepare_data(data, critic, resize=True, shuffle=True):
    print('preparing data...')
    final_dset = []
    high_value_images = []

    num_samples = data.shape[0]

    for _ in range(0, num_samples, BATCH_SIZE):
        # Preprocess
        #images = data[batch_i:batch_i+BATCH_SIZE].astype(np.float32)
        images = data[-BATCH_SIZE:].astype(np.float32)
        images = images.transpose(0, 3, 1, 2) # Transpose observations to be channel-first (BCHW instead of BHWC)
        images /= 255.0 # Normalize observations. Do this here to avoid using too much memory (images are uint8 by default)
        images = Tensor(images).to(torch.device('cpu'))

        data = data[0:-BATCH_SIZE] # to use less memory?

        preds, _ = critic.evaluate(images)
        labels = get_critic_labels(preds)

        images = images.detach().numpy()

        if resize:
            imgs = np.empty((BATCH_SIZE, n_channels, h, h)).astype(np.float32)
            
            for i, image in enumerate(images): # downscale
                image = image.transpose(1, 2, 0) # CHW to HWC
                image = cv2.resize(image, dsize=(h, h))
                image = image.transpose(2, 0, 1) # back to CHW
                imgs[i] = image
        else:
            imgs = images

        # Save (img, label)-tuple for low/high-value images respectively
        if len(final_dset) >= LHV_IMG_COUNT and len(high_value_images) >= LHV_IMG_COUNT:
            break

        if len(final_dset) < LHV_IMG_COUNT:
            final_dset.extend((imgs[i], label) for i, label in enumerate(labels) if label == 0)
        if len(high_value_images) < LHV_IMG_COUNT:
            high_value_images.extend((imgs[i], label) for i, label in enumerate(labels) if label == 1)

    # Make sure enough images were collected
    assert len(final_dset) >= LHV_IMG_COUNT
    assert len(high_value_images) >= LHV_IMG_COUNT

    # Randomize which images get chosen
    #if shuffle:
    #    np.random.shuffle(final_dset)
    #    np.random.shuffle(high_value_images)

    final_dset = final_dset[0:LHV_IMG_COUNT]
    final_dset.extend(high_value_images[0:LHV_IMG_COUNT])
    final_dset = np.array(final_dset, dtype=object)

    del high_value_images
    
    # Randomize order of high and low value images
    if not shuffle:
        np.random.seed(1) # Shuffle high- and low-value-images but keep randomization the same for plotting
    
    np.random.shuffle(final_dset)

    return final_dset

# source: https://github.com/KarolisRam/MineRL2021-Research-baselines/blob/main/standalone/Behavioural_cloning.py#L105
def load_minerl_data(data, shuffle=True):
    print("loading minerl-data...")
    trajectory_names = data.get_trajectory_names()
    if shuffle:
        random.shuffle(trajectory_names)

    all_pov_obs = []
    # Add trajectories to the data until we reach the required DATA_SAMPLES.
    for trajectory_name in trajectory_names:
        trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
        for dataset_observation, _, _, _, _ in trajectory:
            all_pov_obs.append(dataset_observation["pov"])
        if len(all_pov_obs) >= MINERL_SAMPLES:
            dset = np.array(all_pov_obs)
            break

    return dset