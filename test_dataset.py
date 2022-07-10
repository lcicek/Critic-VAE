import pickle
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from vae_utility import load_critic
from vae_parameters import CRITIC_PATH, SAVE_PATH

critic = load_critic(CRITIC_PATH)

with open('/homes/lcicek/Desktop/AAE/recon-dataset.pickle', 'rb') as file:
    recon_dset = pickle.load(file)

dset = np.stack(recon_dset).squeeze()
num_samples = dset.shape[0]

for ep in range(1): # change
        epoch_indices = np.arange(num_samples)

        for batch_i in range(0, num_samples, 1):
            # NOTE: this will cut off incomplete batches from end of the random indices
            batch_indices = epoch_indices[batch_i:batch_i + 1]
            images = dset[batch_indices]
            images = Tensor(images).to(torch.device('cuda:0'))
            
            preds = critic.evaluate(images)

            