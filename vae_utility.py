from io import BytesIO
import os
import minerl
import statistics
import torch
from torch import Tensor
import torch.utils
import torch.distributions
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

from vae_parameters import *
from vae_nets import *

THRESHOLD = 50
font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 10)
titles = ["orig img\n+crit val", "crit val\ninjected", "crit=0\ninjected", "difference\nmask", f"thresholded\nmask\nthr={THRESHOLD}", "ground\ntruth"]

def get_iou(G, T):
    tp = np.sum(G & T) # intersection i.e. true positive
    fn = np.sum(G & np.logical_not(T)) # false negative
    fp = np.sum(np.logical_not(G) & T) # false positive
    
    iou = tp / (tp + fn + fp) # intersection div by union
    
    iou = round(iou, 3)
    fn_rate = round(fn / np.sum(G), 3)
    fp_rate = round(fp / np.sum(np.logical_not(G)), 3)
    
    return iou, fn_rate, fp_rate

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

# source: https://github.com/python-pillow/Pillow/issues/4263
def create_video(trajectory, masks=True):
    print('creating video...')
    byteframes = []
    for f in trajectory[0]:
        byte = BytesIO()
        byteframes.append(byte)
        f.save(byte, format="GIF")
    imgs = [Image.open(byteframe) for byteframe in byteframes]
    imgs[0].save(f"videos/video-threshold={THRESHOLD}.gif", format='GIF', duration=100, save_all=True, loop=0, append_images=imgs[1:])

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
                        title += f"\niou1={ious[0]:03f}\niou2={ious[1]:03f}"
                    draw.text((w*i+2, 0), title, (255,255,255), font=font)

            conc_f.paste(f1, (0, w))
            conc_f.paste(f2, (0, w*2))

            conc_frames.append(conc_f)

        all_conc.append(conc_frames)

    return all_conc

def get_diff_factor(max_values):
    mean_max = statistics.mean(max_values)
    diff_factor = 1.0 / mean_max if mean_max != 0 else 0

    return diff_factor, mean_max

def save_bin_info(bins, bin_frame_count, gt_true_count, gt_mean, second):
    if second:
        num = '2'
    else:
        num = '1'

    total_gt = np.sum(list(gt_true_count.values()))

    with open(f'bin_info_vae{num}.txt', 'w') as f:
        f.write('ground truth pixels sorted by bin:\n')
        for value_bin in gt_true_count:
            count = gt_true_count[value_bin]
            f.write(f'bin: {value_bin}, pixels = {count} = {round(count/total_gt, 2) * 100}%\n')

        f.write('\nground truth mean and std:\n')
        for value_bin in gt_mean:
            mean = round(np.mean(gt_mean[value_bin]), 2)
            std = round(np.nanstd(gt_mean[value_bin]), 2)
            f.write(f'bin: {value_bin}, mean = {mean}, std={std}\n')

        f.write('\nframes separated by bin:\n')
        for value_bin in bin_frame_count:
            count = bin_frame_count[value_bin]
            f.write(f'bin: {value_bin}, frames = {count} = {round(count/1200, 2) * 100}%\n')

        f.write('\nbin-mean and std:\n')
        for value_bin in bins:
            mean = round(np.nanmean(bins[value_bin]), 2)
            std = round(np.nanstd(bins[value_bin]), 2)
            f.write(f'bin: {value_bin}, iou_mean={mean}, iou_std={std}\n')

def eval_textured_frames(trajectory, vae, critic, gt, second=False, t=THRESHOLD):
    print('processing frames...')
    ret = []
    imgs = []
    results = []
    diff_max_values = []
    for i, frame in enumerate(trajectory):
        frame = preprocess_observation(frame)

        preds = critic.evaluate(frame)

        ro, rz, diff, max_value = get_diff_image(vae, frame, preds[0])
        diff_max_values.append(max_value)

        imgs.append([frame, ro, rz, diff, preds[0], gt[i]])

    ious = []
    fn_rates = []
    fp_rates = []
    separated_bins = {}
    separated_bins = defaultdict(lambda: [], separated_bins)
    bin_frame_count = {}
    bin_frame_count = defaultdict(lambda: 0, bin_frame_count)
    gt_true_count = {}
    gt_true_count = defaultdict(lambda: 0, bin_frame_count)
    gt_mean = {}
    gt_mean = defaultdict(lambda: [], bin_frame_count)

    diff_factor, mean_max = get_diff_factor(diff_max_values)

    for img in imgs:
        diff = prepare_diff(img[3], diff_factor, mean_max)
        diff = (diff * 255).astype(np.uint8)
        diff_img = Image.fromarray(diff)

        thresholded = diff > t
        gt = img[5]

        iou, fn_rate, fp_rate = get_iou(gt, thresholded)
        ious.append(iou)
        fn_rates.append(fn_rate)
        fp_rates.append(fp_rate)

        thresh_img = Image.fromarray(thresholded)
        gt_img = Image.fromarray(gt)

        value_bin = round(img[4].item(), 1)
        separated_bins[value_bin].append(iou)
        bin_frame_count[value_bin] += 1
        gt_true_count[value_bin] += gt.sum()
        gt_mean[value_bin].append(gt.sum())

        result_img = save_diff_image(img[0], img[1], img[2], diff_img, img[4], gt_img, thresh_img)
        results.append(result_img)

    save_bin_info(separated_bins, bin_frame_count, gt_true_count, gt_mean, second=second)

    final_iou = np.nanmean(ious)
    final_fn_rate = np.nanmean(fn_rates)
    final_fp_rate = np.nanmean(fp_rates)

    ret.append(results)

    return ret, final_iou, final_fn_rate, final_fp_rate

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

def get_diff_image(autoencoder, img_tensor, pred, one=False):
    if one:
        high_tensor = torch.ones(1).to(device)
    else:
        high_tensor = torch.zeros(1).to(device) + pred

    low_tensor = torch.zeros(1).to(device)

    recon_one = autoencoder.evaluate(img_tensor, high_tensor)
    recon_zero = autoencoder.evaluate(img_tensor, low_tensor)

    recon_one = to_np(recon_one.view(-1, ch, w, w)[0])
    recon_zero = to_np(recon_zero.view(-1, ch, w, w)[0])

    diff = np.subtract(recon_zero, recon_one)
    diff = abs(diff)
    diff = np.transpose(diff, (1, 2, 0))
    diff = np.dot(diff[...,:3], [0.2989, 0.5870, 0.1140]) # to greyscale
    #diff = (diff * 255).astype(np.uint8)
    max_value = np.amax(diff)

    return recon_one, recon_zero, diff, max_value

def prepare_diff(diff_img, diff_factor, mean_max):
    diff_img[diff_img > mean_max] = mean_max
    diff_img = diff_img * diff_factor
    # diff_img = diff_img.astype(np.uint8)

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

def adjust_values(obs):
    img_array = np.array(obs).astype(np.float32)
    img_array /= 255 # to range 0-1
   
    return img_array

def reverse_preprocess(recon):
    recon = to_np(recon.view(-1, ch, w, w)[0])
    recon = recon.transpose(1, 2, 0) # from CHW to HWC
    recon = (recon * 255).astype(np.uint8)

    return recon

def preprocess_observation(obs):
    img_array = adjust_values(obs)
    img_array = img_array.transpose(2, 0, 1) # HWC to CHW for critic
    img_array = img_array[np.newaxis, ...] # add batch_size = 1 to make it BCHW
    img_tensor = Tensor(img_array).to(device)

    return img_tensor

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

def log_info(losses, logger, batch_i, ep, num_samples):
    info = {
        'recon_loss': losses['recon_loss'].item(),
        'kld': losses['KLD'].item(),
        'total_loss': losses['total_loss'].item()
    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, batch_i + (num_samples * ep))

def to_np(x):
    return x.data.cpu().numpy()

def prepare_rgb_image(img_array): # numpy_array
    img_array = np.transpose(img_array, (1, 2, 0)) # CHW to HWC
    img_array = (img_array * 255).astype(np.uint8)
    image = Image.fromarray(img_array, mode='RGB')

    return img_array, image

# source: https://github.com/KarolisRam/MineRL2021-Research-baselines/blob/main/standalone/Behavioural_cloning.py#L105
def load_minerl_data(critic, recon_dset=False, vae=None):
    print("loading minerl-data...")

    ### Initialize mineRL dataset ###
    os.environ['MINERL_DATA_ROOT'] = MINERL_DATA_ROOT_PATH
    data = minerl.data.make('MineRLTreechop-v0', num_workers=1)

    trajectory_names = data.get_trajectory_names()
    rng = np.random.default_rng(seed=0)
    rng.shuffle(trajectory_names)
    
    total_img_count = 75000 if recon_dset else 50000
    collect = 150
    dset = []
    # Add trajectories to the data until we reach the required DATA_SAMPLES.
    for trajectory_name in trajectory_names:
        if len(dset) >= total_img_count:
            break

        print(f'total images = {len(dset)}')
        c_high = 0
        c_mid = 0
        c_low = 0
        trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
        for dataset_observation, _, _, _, _ in trajectory:
            obs = dataset_observation["pov"]
            obs = preprocess_observation(obs)
            pred = critic.evaluate(obs)
            pred = pred[0]

            if recon_dset:
                if pred <= 0.3:
                    continue
                
                obs_high = vae.evaluate(obs, torch.zeros(1).to(device) + pred)                
                obs_low = vae.evaluate(obs, torch.zeros(1).to(device))
                #obs = reverse_preprocess(obs)

                #print(f'memory:: high:{torch.cuda.memory_allocated(obs_high)}, low:{torch.cuda.memory_allocated(obs_low)}, obs: {torch.cuda.memory_allocated(obs)}')

                if c_mid >= collect * 2:
                    break
                else:
                    dset.append(obs_high.detach().cpu().numpy())
                    dset.append(obs_low.detach().cpu().numpy())
                    c_mid += 1
            else:
                obs = obs.detach().cpu().numpy()

                if c_high >= collect and c_low >= collect and c_mid >= collect:
                    break
                elif 0.4 <= pred <= 0.6 and c_mid < collect:
                    dset.append(obs)
                    c_mid += 1
                elif pred >= 0.75 and c_high < collect:
                    dset.append(obs)
                    c_high += 1
                elif pred <= 0.25 and c_low < collect:
                    dset.append(obs)
                    c_low += 1

    #low_val = np.array(dset)
    
    del data # without this line, error gets thrown at the end of the program
    return dset