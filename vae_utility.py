import statistics
import torch
from torch import Tensor
import torch.utils
import torch.distributions
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from collections import defaultdict

from parameters import DATA_SAMPLES
from vae_parameters import *
from vae_nets import *
from utility import load_minerl_data_by_trajectory, prepare_rgb_image, to_np

THRESHOLD = 65
font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 10)
titles = ["orig img\n+crit val", "crit=1\ninjected", "crit=0\ninjected", "difference\nmask", f"thresholded\nmask\nthr={THRESHOLD}", "ground\ntruth"]

def get_iou(A, B):
    intersection = np.sum(A & B)
    union = np.sum(A | B)
    iou = intersection / union
    return round(iou, 3)

def load_textured_minerl():
    evaldatapath = "critic-guided/red-trees/"
    text_dset = np.load(evaldatapath + "X.npy") # / 255.0

    gt_dset = np.expand_dims(np.all(np.load(evaldatapath + "Y.npy"), axis=-1), axis=-1)

    text_dset = text_dset[100:5000:2]
    
    gt_dset = gt_dset[100:5000:2].transpose(0, 3, 1, 2) # gt = ground turth
    gt_dset = gt_dset.squeeze()
    #gt_dset = gt_dset[np.newaxis, ...]
    # Y.transpose(0,3,1,2)

    return text_dset, gt_dset

def create_video(trajectories, masks=True):
    print('creating videos...')
    for i, frames in enumerate(trajectories):
        if masks:
            frames[0].save(f"videos/video-threshold={THRESHOLD}.gif", format='GIF', duration=100, save_all=True, loop=0, append_images=frames[1:])
        else:
            frames[0].save(f"videos/video-{i+1}.gif", format='GIF', duration=100, save_all=True, loop=0, append_images=frames[1:])

def concat_frames(trajs1, trajs2, masks=False, ious=None):
    print('concatting frames...')
    all_conc = []

    for i in range(len(trajs1)):
        conc_frames = []
        frames1 = trajs1[i]
        frames2 = trajs2[i]

        for j in range(len(frames1)):
            f1 = frames1[j] # f = frame
            f2 = frames2[j]

            factor = 6 if masks else 4
            conc_f = Image.new('RGB', (w*factor, w*3))

            if masks:
                draw = ImageDraw.Draw(conc_f)
                for i, title in enumerate(titles):
                    if (i == 5):
                        title += f"\niou1={ious[0]}\niou2={ious[1]}"
                    draw.text((w*i+2, 0), title, (255,255,255), font=font)

            conc_f.paste(f1, (0, w))
            conc_f.paste(f2, (0, w*2))

            conc_frames.append(conc_f)

        all_conc.append(conc_frames)

    return all_conc

def get_diff_factor(max_values):
    mean_max = statistics.mean(max_values)
    diff_factor = 255 // mean_max

    return diff_factor

def save_bin_info(bins, bin_frame_count, second):
    if second:
        num = '1'
    else:
        num = '2'

    with open(f'bin_info_vae{num}.txt', 'w') as f:
        f.write('frames separated by bin:\n')
        for value_bin in bin_frame_count:
            count = bin_frame_count[value_bin]
            f.write(f'bin: {value_bin}, frames = {count} = {round(count/1200, 2) * 100}%\n')

        f.write('\nbin-mean and std:\n')
        for value_bin in bins:
            mean = round(np.nanmean(bins[value_bin]), 2)
            std = round(np.nanstd(bins[value_bin]), 2)
            f.write(f'bin: {value_bin}, iou_mean={mean}, iou_std={std}\n')

def eval_textured_frames(trajectory, vae, critic, gt, second=False):
    print('processing frames...')
    ret = []
    imgs = []
    results = []
    diff_max_values = []
    ious = []

    for i, frame in enumerate(trajectory):
        frame = preprocess_observation(frame)

        preds, _ = critic.evaluate(frame)
        bin_value = round(preds[0].item(), 1)

        ro, rz, diff, max_value = get_diff_image(vae, frame, preds[0])
        diff_max_values.append(max_value)

        imgs.append([frame, ro, rz, diff, preds[0], gt[i]])
    
    diff_factor = get_diff_factor(diff_max_values)

    separated_bins = {}
    separated_bins = defaultdict(lambda: [], separated_bins)
    bin_frame_count = {}
    bin_frame_count = defaultdict(lambda: 0, bin_frame_count)

    for img in imgs:
        diff = prepare_diff(img[3], diff_factor)
        diff_img = Image.fromarray(diff)

        thresholded = diff > THRESHOLD
        gt = img[5]
        iou = get_iou(thresholded, gt)
        ious.append(iou)
        thresh_img = Image.fromarray(thresholded)
        gt_img = Image.fromarray(gt)

        value_bin = round(img[4].item(), 1)
        separated_bins[value_bin].append(iou)
        bin_frame_count[value_bin] += 1

        result_img = save_diff_image(img[0], img[1], img[2], diff_img, img[4], gt_img, thresh_img)
        results.append(result_img)

    save_bin_info(separated_bins, bin_frame_count, second=second)

    final_iou = np.nanmean(ious)
    ret.append(results)

    return ret, final_iou

def evaluate_frames(trajectories, vae, critic):
    print('processing frames...')
    ret = []
    for trajectory in trajectories:
        imgs = []
        results = []
        diff_max_values = []

        for frame in trajectory:
            preds, _ = critic.evaluate(frame)

            ro, rz, diff, max_value = get_diff_image(vae, frame, preds[0])
            diff_max_values.append(max_value)

            imgs.append([frame, ro, rz, diff, preds[0]])
        
        diff_factor = get_diff_factor(diff_max_values)

        for img in imgs:
            diff = prepare_diff(img[3], diff_factor)
            diff_img = Image.fromarray(diff)

            result_img = save_diff_image(img[0], img[1], img[2], diff_img, img[4])
            results.append(result_img)

        ret.append(results)

    return ret

def collect_frames(trajectory_names): # returns list of (64, 64, 3) images for each trajectory
    print('collecting frames...')
    import os
    import minerl
    steps = 1000

    os.environ['MINERL_DATA_ROOT'] = MINERL_DATA_ROOT_PATH
    data = minerl.data.make('MineRLTreechop-v0', num_workers=1)

    all_frames = []
    for name in trajectory_names:
        frames = []

        trajectory = data.load_data(name, skip_interval=0, include_metadata=False)
        for dataset_observation, _, _, _, _ in trajectory:
            obs = dataset_observation["pov"]
            obs = preprocess_observation(obs) # tensor 
            frames.append(obs)

            if len(frames) >= steps:
                all_frames.append(frames)
                break
    
    del data
    return all_frames

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
    if pred > 0.5:
        high_tensor = torch.ones(1).to(device)
    else:
        high_tensor = torch.zeros(1).to(device) + pred

    low_tensor = low_tensor = torch.zeros(1).to(device)

    recon_one = autoencoder.evaluate(img_tensor, high_tensor)
    recon_zero = autoencoder.evaluate(img_tensor, low_tensor)

    recon_one = to_np(recon_one.view(-1, ch, w, w)[0])
    recon_zero = to_np(recon_zero.view(-1, ch, w, w)[0])

    diff = cv2.subtract(recon_zero, recon_one)
    diff = abs(diff)
    diff = np.transpose(diff, (1, 2, 0))
    diff = np.dot(diff[...,:3], [0.2989, 0.5870, 0.1140]) # to greyscale
    diff = (diff * 255).astype(np.uint8)
    max_value = np.amax(diff)

    return recon_one, recon_zero, diff, max_value

def prepare_diff(diff_img, diff_factor):
    diff_img *= diff_factor
    diff_img[diff_img > 255] = 255

    return diff_img

def save_diff_image(img_tensor, recon_one, recon_zero, diff_img, pred, gt_img=None, thresh_img=None):
    conc_h = np.array(np.concatenate((
        to_np(img_tensor.view(-1, ch, w, w)[0]),
        recon_one,
        recon_zero,
    ), axis=2))

    _, conc_img = prepare_rgb_image(conc_h)        
    
    factor = 4 if gt_img is None else 6

    img = Image.new('RGB', (w*factor, w))
    img.paste(conc_img, (0, 0))
    img.paste(diff_img, (w*3, 0))
    if factor == 6:
        img.paste(thresh_img, (w*4, 0))
        img.paste(gt_img, (w*5, 0))

    draw = ImageDraw.Draw(img)
    draw.text((2, 2), f'{pred.item():.1f}', (255,255,255), font=font)

    return img

def preprocess_observation(obs):
    img_array = np.array(obs).astype(np.float32)
    img_array = img_array.transpose(2, 0, 1) # HWC to CHW for critic
    img_array = img_array[np.newaxis, ...] # add batch_size = 1 to make it BCHW
    img_array /= 255 # to range 0-1
    img_tensor = Tensor(img_array).to(device)

    return img_tensor

def create_recon_dataset(vae, critic):
    traj_dict = load_minerl_data_by_trajectory()

    for traj_name in traj_dict:
        print(f'trajectory: {traj_name}')
        recons = []
        povs = traj_dict[traj_name]

        for pov in povs:
            img_tensor = preprocess_observation(pov)

            preds, _ = critic.evaluate(img_tensor)
            recon = vae.evaluate(img_tensor, preds[0])

            # revert preprocessing to save image later on
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