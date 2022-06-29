# SOURCE: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
import torch #; torch.manual_seed(0)
from torch import Tensor, t
import torch.utils
import torch.distributions
import numpy as np
import os
from PIL import Image
from time import time
import argparse
import os
import pickle
import statistics

from vae_parameters import *
from vae_nets import *
from utility import initialize, prepare_data
from vae_utility import *
from logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument('-t', action='store_true') # train
parser.add_argument('-i', action='store_true') # show recons of samples
parser.add_argument('-dataset', action='store_true') # save recons as dataset
parser.add_argument('-second', action='store_true') # train second VAE
parser.add_argument('-evalsecond', action='store_true')
parser.add_argument('-video', action='store_true')
parser.add_argument('-masks', action='store_true')
args = parser.parse_args()

TRAIN = args.t
INJECT = args.i
CREATE_DATASET = args.dataset
TRAIN_SECOND_VAE = args.second
EVAL_SECOND_VAE = args.evalsecond
VIDEO = args.video
MASKS = args.masks

def train(autoencoder, dset, logger):
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
                log_info(losses, logger, batch_i, ep)

    return autoencoder

def image_evaluate(autoencoder, critic):
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
            diff_img = prepare_diff(img[2], diff_factor)
            diff_img = Image.fromarray(diff_img)
            save_img = save_diff_image(img_tensor, img[0], img[1], diff_img, preds[0])

            save_img.save(f'{SAVE_PATH}/image-{i:03d}.png', format="png")
    

vae = VariationalAutoencoder().to(device) # GPU


if True:
    # get images from regular vae
    load_vae_network(vae)
    critic = load_critic(CRITIC_PATH)

    if True:
        frames, gt_frames = load_textured_minerl() # gt = ground truth of tree trunk
    else:
        trajectory_names = [
            "v3_content_squash_angel-3_16074-17640",
            "v3_smooth_kale_loch_ness_monster-1_4439-6272",
            "v3_cute_breadfruit_spirit-6_17090-19102",
            "v3_key_nectarine_spirit-2_7081-9747",
        ]
        frames = collect_frames(trajectory_names)

    vae_frames, iou1 = evaluate_frames(frames, vae, critic, textured=True, gt=gt_frames)

    # get images from second vae
    vae = VariationalAutoencoder().to(device) # new
    load_vae_network(vae, second_vae=True)
    critic = load_critic(SECOND_CRITIC_PATH)

    second_vae_frames, iou2 = evaluate_frames(frames, vae, critic, textured=True, gt=gt_frames)

    concatenated = concat_frames(vae_frames, second_vae_frames, masks=True, ious=(iou1, iou2))
    create_video(concatenated)

elif CREATE_DATASET:
    load_vae_network(vae)
    critic = load_critic(CRITIC_PATH)
    dataset = create_recon_dataset(vae, critic)

    with open(SAVE_DATASET_PATH, 'wb') as file:
        pickle.dump(dataset, file)
elif TRAIN_SECOND_VAE:
    critic = load_critic(SECOND_CRITIC_PATH)

    print('preparing dataset...')
    with open('/homes/lcicek/Desktop/AAE/recon-dataset.pickle', 'rb') as file:
        recon_dset = pickle.load(file)

    recon_dset = prepare_recon_dataset(recon_dset)
    recon_dset = prepare_data(recon_dset, critic, resize=False)

    logger = Logger('./logs/vae' + str(time())[-5::])
    vae = train(vae, recon_dset, logger)

    torch.save(vae.encoder.state_dict(), SECOND_ENCODER_PATH)
    torch.save(vae.decoder.state_dict(), SECOND_DECODER_PATH)
elif EVAL_SECOND_VAE:
    critic = load_critic(SECOND_CRITIC_PATH)
    load_vae_network(vae, second_vae=True)
    image_evaluate(vae, critic)
else: # REGULAR VAE
    dset, critic = initialize()

    if TRAIN:
        logger = Logger('./logs/vae' + str(time())[-5::])
        vae = train(vae, dset, logger)

        torch.save(vae.encoder.state_dict(), ENCODER_PATH)
        torch.save(vae.decoder.state_dict(), DECODER_PATH)
    else: # EVALUATE
        load_vae_network(vae)
        image_evaluate(vae, critic)