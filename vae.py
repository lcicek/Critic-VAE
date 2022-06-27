# SOURCE: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
import torch #; torch.manual_seed(0)
from torch import Tensor, t
import torch.utils
import torch.distributions
import numpy as np
import os
from PIL import Image, ImageDraw, ImageOps
from time import time
import argparse
import cv2
import os
import pickle
import statistics

from parameters import DATA_SAMPLES, CRITIC_PATH
from vae_parameters import *
from vae_nets import *
from utility import initialize, load_minerl_data_by_trajectory, prepare_rgb_image, to_np, prepare_data
from logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument('-t', action='store_true') # train
parser.add_argument('-i', action='store_true') # show recons of samples
parser.add_argument('-dataset', action='store_true') # save recons as dataset
parser.add_argument('-second', action='store_true') # train second VAE
parser.add_argument('-evalsecond', action='store_true')
args = parser.parse_args()

TRAIN = args.t
INJECT = args.i
DATASET = args.dataset
SECOND = args.second
EVAL_SECOND = args.evalsecond

def train(autoencoder, dset):
    opt = torch.optim.Adam(autoencoder.parameters(), lr=lr) 
    num_samples = dset.shape[0]

    # Start training
    for ep in range(epochs): # change
        epoch_indices = np.arange(num_samples)
        np.random.shuffle(epoch_indices)

        print(f'epoch: {ep}')

        for batch_i in range(0, num_samples, batch_size):
            # NOTE: this will cut off incomplete batches from end of the random indices
            batch_indices = epoch_indices[batch_i:batch_i + batch_size]

            all_data = dset[batch_indices]
            
            images = np.array([d[0] for d in all_data])
            images = Tensor(images).to(device)

            preds, _ = critic.evaluate(images)

            opt.zero_grad()
            out = autoencoder(images, preds)
            losses = autoencoder.vae_loss(out[0], out[1], out[2], out[3])
            loss = losses['total_loss']
            loss.backward()
            opt.step()

            if batch_i % log_n == 0:
                print(f'step {batch_i + (DATA_SAMPLES * ep)}')

                info = {
                    'recon_loss': losses['recon_loss'].item(),
                    'kld': losses['KLD'].item(),
                    'total_loss': losses['total_loss'].item()
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, batch_i + (DATA_SAMPLES * ep))

    return autoencoder

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

def image_results(autoencoder, critic):
    folder = os.listdir(EVAL_IMAGES_PATH)
    imgs = []
    diff_max_values = []
    for i, img_file in enumerate(folder):
        ### LOAD IMAGES AND PREPROCESS ###
        orig_img = Image.open(f'{EVAL_IMAGES_PATH}/{img_file}')
        img_array = np.array(orig_img).astype(np.float32)
        img_array = img_array.transpose(2, 0, 1) # HWC to CHW for critic
        img_array = img_array[np.newaxis, ...] # add batch_size = 1 to make it BCHW
        img_array /= 255 # to range 0-1
        img_tensor = Tensor(img_array).to(device)

        preds, _ = critic.evaluate(img_tensor)

        if INJECT:
            img = get_injected_img(autoencoder, img_tensor, preds[0])
            img.save(f'{SAVE_PATH}/image-{i:03d}.png', format="png")
        else:
            ro, rz, diff, max_value = get_diff_image(autoencoder, img_tensor, preds[0])
            imgs.append([ro, rz, diff])
            diff_max_values.append(max_value)
    
    if not INJECT:
        mean_max = statistics.mean(diff_max_values)
        diff_factor = 255 // mean_max

        for i, img in enumerate(imgs):
            diff_img = prepare_diff_image(img[2], diff_factor)
            save_img = save_diff_image(img_tensor, img[0], img[1], diff_img, preds[0])

            save_img.save(f'{SAVE_PATH}/image-{i:03d}.png', format="png")

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

def preprocess(recon_dset):
    recon_dset = list(recon_dset.values())
    ret = []

    for traj in recon_dset:
        ret.append(np.array(traj))
        #ret.append(np.array(traj).astype(np.float32).transpose(0, 3, 1, 2) / 255)
    
    ret = np.concatenate(ret, axis=0)

    return ret

if DATASET:
    vae = VariationalAutoencoder().to(device) # GPU

    try:
        vae.encoder.load_state_dict(torch.load(ENCODER_PATH))
        vae.decoder.load_state_dict(torch.load(DECODER_PATH))
    except Exception as e:
        print(e)
    
    vae.eval()
    vae.encoder.eval()
    vae.decoder.eval()

    from critic_net import Critic
    critic = Critic()
    critic.load_state_dict(torch.load(CRITIC_PATH))
    critic.to(device)
    critic.eval()

    dataset = create_recon_dataset(vae, critic)

    with open(SAVE_DATASET_PATH, 'wb') as file:
        pickle.dump(dataset, file)
elif SECOND:
    from critic_net import Critic
    critic = Critic()
    critic.load_state_dict(torch.load(SECOND_CRITIC_PATH))
    critic.eval()

    print('preparing dataset...')
    with open('/homes/lcicek/Desktop/AAE/recon-dataset.pickle', 'rb') as file:
        recon_dset = pickle.load(file)

    recon_dset = preprocess(recon_dset)
    recon_dset = prepare_data(recon_dset, critic, resize=False)
    critic.to(device)

    print('starting training...')
    logger = Logger('./logs/vae' + str(time())[-5::])
    vae = VariationalAutoencoder().to(device) # GPU
    vae = train(vae, recon_dset)

    torch.save(vae.encoder.state_dict(), SECOND_ENCODER_PATH)
    torch.save(vae.decoder.state_dict(), SECOND_DECODER_PATH)
elif True:
    from critic_net import Critic
    critic = Critic()
    critic.load_state_dict(torch.load(SECOND_CRITIC_PATH))
    critic.eval()
    critic.to(device)

    vae = VariationalAutoencoder().to(device) # GPU

    try:
        vae.encoder.load_state_dict(torch.load(SECOND_ENCODER_PATH))
        vae.decoder.load_state_dict(torch.load(SECOND_DECODER_PATH))
    except Exception as e:
        print(e)
    
    vae.eval()
    vae.encoder.eval()
    vae.decoder.eval()

    image_results(vae, critic)

else:
    dset, critic = initialize()
    vae = VariationalAutoencoder().to(device) # GPU

    if TRAIN:
        logger = Logger('./logs/vae' + str(time())[-5::])
        vae = train(vae, dset)

        torch.save(vae.encoder.state_dict(), ENCODER_PATH)
        torch.save(vae.decoder.state_dict(), DECODER_PATH)
    else:
        try:
            vae.encoder.load_state_dict(torch.load(ENCODER_PATH))
            vae.decoder.load_state_dict(torch.load(DECODER_PATH))
        except Exception as e:
            print(e)
        
        vae.eval()
        vae.encoder.eval()
        vae.decoder.eval()

        image_results(vae, critic)