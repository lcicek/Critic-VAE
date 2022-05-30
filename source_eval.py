import torch
from torch import Tensor
import os
import numpy as np
from PIL import Image
from nets import *
from parameters import *
from utility import prepare_rgb_image, get_critic_labels, prior_sample, to_np
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### LOAD NETWORKS ### 
critic = Critic().to(device)
critic.load_state_dict(torch.load(CRITIC_PATH, map_location=device))

Q = Q_net(X_dim=n_channels, N=n, z_dim=z_dim).to(device)
P = P_net(X_dim=n_channels, N=n, z_dim=z_dim).to(device)
D = D_net(z_dim, z_dim).to(device)

try:
    Q.load_state_dict(torch.load('Q_encoder_weights.pt'))
    P.load_state_dict(torch.load('P_decoder_weights.pt'))
    D.load_state_dict(torch.load('D_discriminator_weights.pt'))
except Exception as e:
    print(e) 

Q.eval()
P.eval()
D.eval()
critic.eval()

folder = os.listdir(EVAL_IMAGES_PATH)
for i, img_file in enumerate(folder):
        orig_img = Image.open(f'{EVAL_IMAGES_PATH}/{img_file}')
        img_array = np.array(orig_img).astype(np.float32)
        img_array = img_array.transpose(2, 0, 1) # HWC to CHW for critic
        img_array = img_array[np.newaxis, ...]
        img_array /= 255
        img_tensor = Tensor(img_array).to(device)

        preds, _ = critic.evaluate(img_tensor)
        label = get_critic_labels(preds).item()

        img_array = np.squeeze(img_array)
        img_array = cv2.resize(img_array.transpose(1, 2, 0), dsize=(16, 16)) # CHW to HWC for resize
        img_array = img_array.transpose(2, 0, 1) # back to CHW for Q and P
        img_array = img_array[np.newaxis, ...]
        img_tensor = Tensor(img_array).to(device)

        class_out, z_sample = Q(img_tensor)
        X_sample = P(z_sample)

        D_fake = D(z_sample)[0].item()
        D_real = D(prior_sample(label).unsqueeze(0))[0].item()

        zero_prob = class_out[0][0].item()
        one_prob = class_out[0][1].item()
        class_error = one_prob if label == 0 else zero_prob

        ps = torch.rand(z_dim).to(device) # random values
        ps_prob1 = (ps.unsqueeze(0))[0][0].item()

        print(f'img: {i:03d}, class-error: {class_error:.2f}, d-fake: {D_fake:.2f}, d-real: {D_real:.2f}, random: {ps_prob1:.2f}')

        conc_h = np.concatenate((to_np(img_tensor.view(-1, 3, h, h)[0]), to_np(X_sample.view(-1, 3, h, h)[0])), axis=2)
        _, img = prepare_rgb_image(conc_h)

        img.save(f'{SAVE_PATH}/image-{i:03d}-label={label}.png', format="png")

print('aim for these values: d-fake = d-real = 1, random != 1')