import torch
from torch import Tensor
import torch.utils
import torch.distributions
import numpy as np
from PIL import Image, ImageDraw
import cv2

from parameters import DATA_SAMPLES
from vae_parameters import *
from vae_nets import *
from utility import load_minerl_data_by_trajectory, prepare_rgb_image, to_np

def get_injected_img(autoencoder, img_tensor, pred):
    orig_recon = autoencoder.evaluate(img_tensor, pred)
    recons = autoencoder.inject(img_tensor)

    conc_h = np.concatenate((
        to_np(img_tensor.view(-1, ch, w, w)[0]),
        to_np(orig_recon.view(-1, ch, w, w)[0]),
    ), axis=2)

    conc_recons = np.concatenate([to_np(recons[i].view(-1, ch, w, w)[0]) for i in range(inject_n)], axis=2)
    conc_h = np.concatenate((conc_h, conc_recons), axis=2)

    _, img = prepare_rgb_image(conc_h)

    return img

def get_diff_image(autoencoder, img_tensor, pred):
    #if pred < 0.6: # see if we can amplify trees in high value images
    #    continue # skip low value images
    
    recon_one = autoencoder.evaluate(img_tensor, torch.ones(1).to(device))
    recon_zero = autoencoder.evaluate(img_tensor, torch.zeros(1).to(device))

    recon_one = to_np(recon_one.view(-1, ch, w, w)[0])
    recon_zero = to_np(recon_zero.view(-1, ch, w, w)[0])

    diff = cv2.subtract(recon_zero, recon_one)
    diff = abs(diff)
    diff = np.transpose(diff, (1, 2, 0))
    diff = np.dot(diff[...,:3], [0.299, 0.587, 0.114]) # to greyscale
    diff = (diff * 255).astype(np.uint8)
    max_value = np.amax(diff)

    return recon_one, recon_zero, diff, max_value

def prepare_diff_image(diff_img, diff_factor):
    diff_img *= diff_factor
    diff_img[diff_img > 255] = 255
    diff_img = Image.fromarray(diff_img)

    return diff_img

def save_diff_image(img_tensor, recon_one, recon_zero, diff_img, pred):
    conc_h = np.array(np.concatenate((
        to_np(img_tensor.view(-1, ch, w, w)[0]),
        recon_one,
        recon_zero,
    ), axis=2))

    _, conc_img = prepare_rgb_image(conc_h)        
    
    img = Image.new('RGB', (w*4, w))
    img.paste(conc_img, (0, 0))
    img.paste(diff_img, (w*3, 0))

    draw = ImageDraw.Draw(img)
    draw.text((2, 2), f'{pred.item():.1f}', (255,255,255))

    return img

def create_recon_dataset(vae, critic):
    traj_dict = load_minerl_data_by_trajectory()

    for traj_name in traj_dict:
        print(f'trajectory: {traj_name}')
        recons = []
        povs = traj_dict[traj_name]

        for pov in povs:
            img_array = np.array(pov).astype(np.float32)
            img_array = img_array.transpose(2, 0, 1) # HWC to CHW for critic
            img_array = img_array[np.newaxis, ...] # add batch_size = 1 to make it BCHW
            img_array /= 255 # to range 0-1
            img_tensor = Tensor(img_array).to(device)

            preds, _ = critic.evaluate(img_tensor)
            recon = vae.evaluate(img_tensor, preds[0])

            recon = to_np(recon.view(-1, ch, w, w)[0])
            recon = recon.transpose(1, 2, 0) # from CHW to HWC
            recon *= 255
            recon = recon.astype(np.uint8)
            recons.append(recon)
        
        traj_dict[traj_name] = recons # update value

    return traj_dict

def prepare_recon_dataset(recon_dset):
    recon_dset = list(recon_dset.values())
    ret = []

    for traj in recon_dset:
        ret.append(np.array(traj))
    
    ret = np.concatenate(ret, axis=0)

    return ret

def load_vae_network(vae, second_vae=False):
    if second_vae:
        enc_path = SECOND_ENCODER_PATH
        dec_path = SECOND_DECODER_PATH
    else:
        enc_path = ENCODER_PATH
        dec_path = DECODER_PATH

    try:
        vae.encoder.load_state_dict(torch.load(enc_path))
        vae.decoder.load_state_dict(torch.load(dec_path))
    except Exception as e:
        print(e)

    vae.eval()
    vae.encoder.eval()
    vae.decoder.eval()

def load_critic(path):
    from critic_net import Critic
    critic = Critic()
    critic.load_state_dict(torch.load(path))
    critic.eval()
    critic.to(device)

    return critic

def log_info(losses, logger, batch_i, ep):
    print(f'step {batch_i + (DATA_SAMPLES * ep)}')

    info = {
        'recon_loss': losses['recon_loss'].item(),
        'kld': losses['KLD'].item(),
        'total_loss': losses['total_loss'].item()
    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, batch_i + (DATA_SAMPLES * ep))