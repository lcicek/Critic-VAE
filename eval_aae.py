import torch
from torch import Tensor
import os
import numpy as np
from PIL import Image
from nets import Q_net, P_net, Critic
from parameters import *
from utility import prepare_rgb_image, get_critic_labels, to_np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### LOAD NETWORKS ### 
critic = Critic().to(device)
critic.load_state_dict(torch.load(CRITIC_PATH, map_location=device))

Q = Q_net(X_dim=n_channels, N=n, z_dim=z_dim).to(device)
shape = Q.get_shape(torch.zeros((BATCH_SIZE, n_channels, h, h)).to(device))
P = P_net(X_dim=n_channels, N=n, z_dim=z_dim, inner_shape=shape).to(device)

try:
    Q.load_state_dict(torch.load('Q_encoder_weights_RCD.pt')) # change
    P.load_state_dict(torch.load('P_decoder_weights_RCD.pt'))
except Exception as e:
    print(e) 

Q.eval()
P.eval()

folder = os.listdir(EVAL_IMAGES_PATH)
for i, img_file in enumerate(folder):
        orig_img = Image.open(f'{EVAL_IMAGES_PATH}/{img_file}')
        img_array = np.array(orig_img).astype(np.float32)
        img_array = img_array.transpose(2, 0, 1) # CHW
        img_array = img_array[np.newaxis, ...]
        img_array /= 255
        img_tensor = Tensor(img_array).to(device)

        preds, _ = critic.evaluate(img_tensor)
        label = get_critic_labels(preds).item()

        class_out, z_sample = Q(img_tensor)
        x_sample = P(z_sample)

        print(f'class: {Tensor.tolist(class_out[0])}, label: {label}')

        conc_h = np.concatenate((to_np(img_tensor.view(-1, 3, h, h)[0]), to_np(x_sample.view(-1, 3, h, h)[0])), axis=2)
        _, img = prepare_rgb_image(conc_h)

        img.save(f'{SAVE_PATH}/image-{i:03d}-label={label}.png', format="png")

"""
def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
"""