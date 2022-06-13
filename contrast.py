import cv2
import os
import minerl
import torch
import numpy as np
from torch import Tensor
from PIL import Image

from parameters import MINERL_DATA_ROOT_PATH, CRITIC_PATH, device
from nets import *
from utility import load_minerl_data, prepare_rgb_image, to_np

CROP_PATH = '/homes/lcicek/Desktop/AAE/crop'
data_samples = 10

class ContrastiveEncoder(nn.Module):
    def __init__(self, dims=[64, 128, 256, 512, 1024]):
        super(ContrastiveEncoder, self).__init__()

        self.model = nn.Sequential(
                        nn.Conv2d(3, dims[0], 5, 1, 2), # to 16x16x64
                        nn.BatchNorm2d(dims[0]),
                        nn.MaxPool2d(2), # to 8x8x64
                        nn.ReLU(),

                        nn.Conv2d(dims[0], dims[1], 5, 1, 2), # to 8x8x128
                        nn.BatchNorm2d(dims[1]),
                        nn.MaxPool2d(2), # to 4x4x128
                        nn.ReLU(),
                        
                        nn.Conv2d(dims[1], dims[2], 5, 1, 2), # to 4x4x256
                        nn.BatchNorm2d(dims[2]),
                        nn.MaxPool2d(2), # to 2x2x256
                        nn.ReLU(),
                        
                        nn.Conv2d(dims[2], dims[3], 5, 1, 2), # to 2x2x512
                        nn.BatchNorm2d(dims[3]),
                        nn.MaxPool2d(2), # to 1x1x512
                        nn.ReLU(),

                        nn.Conv2d(dims[2], dims[3], 5, 1, 2), # to 1x1x1024
                        nn.BatchNorm2d(dims[3]),
                        nn.ReLU(),
                    )
        
        self.class_output = nn.Linear(dims[4], 1) # one prediction value

    def forward(self, x):
        for layer in self.model:
            x = layer(x)

        class_out = self.class_output(torch.flatten(x, start_dim=1))
        
        return x, class_out

def save_images(original, images, preds):
    original = Image.fromarray(original, mode='RGB')
    original.save(f'{CROP_PATH}/original.png', format="png")

    for i, image in enumerate(images):
        _, img = prepare_rgb_image(to_np(image))
        img.save(f'{CROP_PATH}/image-{i:03d}-p={preds[i].item()}.png', format="png")

def transform(img): # img: (batch_size=1, channel, width, height)
    res = []
    w = 16

    for r in range(0, 64, w): # row
        for c in range(0, 64, w): # column
            crop = img[r:r+w, c:c+w, :]
            resized = cv2.resize(crop, (64, 64))
            res.append(resized)

    return res

def train(encoder, images, preds):
    optim_enc = torch.optim.Adam(encoder.parameters(), lr=0.001)

    return -1


### Initialize mineRL dataset ###
os.environ['MINERL_DATA_ROOT'] = MINERL_DATA_ROOT_PATH
data = minerl.data.make('MineRLTreechop-v0', num_workers=1)
all_pov_obs = load_minerl_data(data) # get all minerl observations
del data # without this line, error gets thrown at the end of the program

critic = Critic().to(device)
critic.load_state_dict(torch.load(CRITIC_PATH))
critic.eval()

encoder = ContrastiveEncoder().to(device)

all_pov_obs = all_pov_obs[:data_samples]
dset = np.empty([data_samples, 16, 64, 64])
for i, entry in enumerate(all_pov_obs):
    images = transform(entry)
    images = np.transpose(images, (0, 3, 1, 2)).astype(np.float32) / 255
    dset[i] = images

dset = Tensor(dset).to(device)
#original = Image.fromarray(entry, mode='RGB')
#original.save(f'{CROP_PATH}/original.png', format="png")

#for j, image in enumerate(images):
#    _, img = prepare_rgb_image(to_np(image))
#    img.save(f'{CROP_PATH}/image-{i}-{j:03d}-p={preds[j].item()}.png', format="png")


# save_images(entry, images, preds)

# encoder = ContrastiveEncoder().to(device)