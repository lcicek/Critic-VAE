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
import denseCRF

from vae_parameters import *
from vae_nets import *

THRESHOLD = 50
font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 10)
titles = ["orig img\n+crit val", "crit val\ninjected", "crit=0\ninjected", "difference\nmask", f"thr-mask\nthr={THRESHOLD}", "thr-mask +\ncrf", "ground\ntruth"]

# copied from critic-code, source and github link are listed in bachelor thesis
def crf(imgs, mask, Y, skip=1):
        mask = mask.copy()
        
        w1    = [22]   # weight of bilateral term
        alpha = [12]   # spatial std
        beta  = [3.1]  # rgb  std
        w2    = [8]    # weight of spatial term
        gamma = [1.8]  # spatial std
        it    = [10]   # iteration
        res = []
        params = []
        for param in [(a,b,c,d,e,i) for a in w1 for b in alpha for c in beta for d in w2 for e in gamma for i in it]:
            M = mask[::skip]
            #param = (w1, alpha, beta, w2, gamma, it)
            for i, img in enumerate(imgs[::skip]):
                maskframe = M[i,0]
                prob = np.stack((1-maskframe, maskframe), axis=-1)
                seg = denseCRF.densecrf(img, prob, param)
                
                M[i,0] = seg
            M = M.transpose(0, 2, 3, 1).astype(np.bool)
           
            r = np.sum(Y[::skip] & M)/np.sum(Y[::skip] | M)
            res.append(r)
            params.append(param)

        res = np.array(res)
        order = np.argsort(res)
        res = res[order]
        params = np.array(params)[order]

        mask[::skip] = M.transpose(0,3,1,2)
        return (mask >= 1)

def get_iou(G, T):
    tp = np.sum(G & T) # intersection i.e. true positive
    fn = np.sum(G & np.logical_not(T)) # false negative
    fp = np.sum(np.logical_not(G) & T) # false positive
    
    if tp+fn+fp == 0: # 0 out of 0 correctly classified pixels is equivalent to IoU=1
        iou = 1 # so avoid dividing by zero
    else:
        iou = tp / (tp + fn + fp) # intersection div by union
    
    iou = round(iou, 3)
    
    return iou

def load_textured_minerl():
    text_dset = np.load(MINERL_EPISODE_PATH + "X.npy") # / 255.0

    gt_dset = np.expand_dims(np.all(np.load(MINERL_EPISODE_PATH + "Y.npy"), axis=-1), axis=-1)

    text_dset = text_dset[100:5000:2]
    
    gt_dset = gt_dset[100:5000:2].transpose(0, 3, 1, 2) # gt = ground turth
    gt_dset = gt_dset.squeeze()
    #gt_dset = gt_dset[np.newaxis, ...]
    # Y.transpose(0,3,1,2)

    return text_dset, gt_dset

# source: https://github.com/python-pillow/Pillow/issues/4263
def create_video(frames):
    #count = 200
    #for i, frame in enumerate(frames): # save first "count" images
    #    if i > count:
    #        break

    #    frame.save(f'{SAVE_PATH}/image-{i:03d}.png', format="png")

    print('creating video...')

    if not os.path.exists(VIDEO_PATH):
        os.mkdir(VIDEO_PATH)

    byteframes = []
    for f in frames:
        byte = BytesIO()
        byteframes.append(byte)
        f.save(byte, format="GIF")
    imgs = [Image.open(byteframe) for byteframe in byteframes]
    imgs[0].save(f"{VIDEO_PATH}video-threshold={THRESHOLD}.gif", format='GIF', duration=100, save_all=True, loop=0, append_images=imgs[1:])

def get_diff_factor(max_values):
    mean_max = statistics.mean(max_values)
    diff_factor = 1.0 / mean_max if mean_max != 0 else 0

    return diff_factor, mean_max

def save_bin_info_file(bin_ious, bin_frames, bin_gts):
    total_gt = np.sum(list(bin_gts.values()))

    with open(f'bin_info_vae1.txt', 'w') as f:
        f.write('ground truth pixels sorted by bin:\n')
        for value_bin in bin_gts:
            count = bin_gts[value_bin]
            f.write(f'bin: {value_bin}, pixels = {count} = {round(count/total_gt, 2) * 100}%\n')

        f.write('\nframes separated by bin:\n')
        for value_bin in bin_frames:
            count = bin_frames[value_bin]
            f.write(f'bin: {value_bin}, frames = {count} = {round(count/1200, 2) * 100}%\n')

        f.write('\niou-mean and std:\n')
        for value_bin in bin_ious:
            mean = round(statistics.mean(bin_ious[value_bin]), 2)
            std = round(statistics.stdev(bin_ious[value_bin]), 2)
            f.write(f'bin: {value_bin}, iou_mean={mean}, iou_std={std}\n')

def save_bin_info(preds, gt, thr_masks):
    bin_ious = defaultdict(lambda: [], {})
    bin_frames = defaultdict(lambda: 0, {})
    bin_gts = defaultdict(lambda: 0, {})

    for i, pred in enumerate(preds):
        value_bin = round(pred.item(), 1)
        thr_iou = get_iou(thr_masks[i], gt[i])

        bin_ious[value_bin].append(thr_iou)
        bin_frames[value_bin] += 1
        bin_gts[value_bin] += gt[i].sum()
    
    save_bin_info_file(bin_ious, bin_frames, bin_gts)


def get_diff_and_thr_masks(diff_masks, max_values, thr=THRESHOLD):
    thr_masks = []

    diff_factor, mean_max = get_diff_factor(max_values)
    for i, diff in enumerate(diff_masks):
        diff = prepare_diff(diff, diff_factor, mean_max)
        diff = (diff * 255).astype(np.uint8)

        thr_mask = diff > thr
        thr_masks.append(thr_mask)
        diff_masks[i] = diff
    
    return np.array(diff_masks), np.array(thr_masks)

def eval_textured_frames(trajectory, vae, critic, gt, t=THRESHOLD):
    print('processing frames...')
    one_recons = []
    zero_recons = []
    diff_masks = [] # unnormalized yet
    preds = []
    frames = []
    max_values = []

    for image in trajectory:
        frame = preprocess_observation(image)
        pred = critic.evaluate(frame)
        ro, rz, diff, max_value = get_diff_image(vae, frame, pred[0])

        one_recons.append(ro)
        zero_recons.append(rz)
        diff_masks.append(diff)
        max_values.append(max_value)
        preds.append(pred[0])
        frames.append(frame)

    diff_masks, thr_masks = get_diff_and_thr_masks(diff_masks, max_values, thr=t)
    thr_iou = get_iou(gt, thr_masks)

    crf_imgs = trajectory[:, np.newaxis, ...]
    crf_diff_mask = np.array(thr_masks)[:, np.newaxis, ...].astype(np.float32)
    crf_gt = gt[..., np.newaxis]

    crf_masks = crf(crf_imgs, crf_diff_mask, crf_gt).squeeze()
    crf_iou = get_iou(gt, crf_masks)

    ret = []
    for i, frame in enumerate(frames):
        final_frame = get_final_frame(
            frame,
            one_recons[i],
            zero_recons[i],
            Image.fromarray(diff_masks[i]),
            preds[i],
            gt_img=Image.fromarray(gt[i]),
            thr_img=Image.fromarray(thr_masks[i]),
            crf_img=Image.fromarray(crf_masks[i]),
            thr_iou=thr_iou,
            crf_iou=crf_iou
        )

        ret.append(final_frame)

    save_bin_info(preds, gt, thr_masks)

    return ret, thr_iou, crf_iou

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
        #to_np(orig_recon.view(-1, ch, w, w)[0]),
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

def get_final_frame(img_tensor, recon_one, recon_zero, diff_img, pred, gt_img=None, thr_img=None, crf_img=None, thr_iou=None, crf_iou=None):
    conc_h = np.array(np.concatenate((
        to_np(img_tensor.view(-1, ch, w, w)[0]),
        recon_one,
        recon_zero,
    ), axis=2))

    _, conc_img = prepare_rgb_image(conc_h)        
    
    with_masks = gt_img is not None

    image_count = 7 if with_masks else 4
    height = w*2 if with_masks else w
    ih = w if with_masks else 0 # image height
    width = w*image_count

    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    img.paste(conc_img, (0, ih))
    img.paste(diff_img, (w*3, ih))
    if with_masks:
        img.paste(thr_img, (w*4, ih))
        img.paste(crf_img, (w*5, ih))
        img.paste(gt_img, (w*6, ih))

        for i, title in enumerate(titles):
            if (i == 4):
                title += f"\niou={thr_iou}"
            elif (i == 5):
                title += f"\niou={crf_iou}"
                
            draw.text((w*i+2, 0), title, (255,255,255), font=font)

    draw = ImageDraw.Draw(img)
    draw.text((2, ih+2), f'{pred.item():.1f}', (255,255,255), font=font)

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
    
    collect = 150
    dset = []
    # Add trajectories to the data until we reach the required DATA_SAMPLES.
    for trajectory_name in trajectory_names:
        if len(dset) >= total_images: # total_images defined in vae_parameters.py
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
                obs_pred = vae.evaluate(obs, torch.zeros(1).to(device) + pred)                
                obs_low = vae.evaluate(obs, torch.zeros(1).to(device))

                obs_pred = obs_pred.detach().cpu().numpy()
                obs_low = obs_low.detach().cpu().numpy()

                #print(f'memory:: high:{torch.cuda.memory_allocated(obs_high)}, low:{torch.cuda.memory_allocated(obs_low)}, obs: {torch.cuda.memory_allocated(obs)}')

                if c_high >= collect and c_low >= collect and c_mid >= collect:
                    break
                elif 0.4 <= pred <= 0.6 and c_mid < collect:
                    dset.append(obs_pred)
                    dset.append(obs_low)
                    c_mid += 1
                elif pred >= 0.7 and c_high < collect:
                    dset.append(obs_pred)
                    c_high += 1
                elif pred <= 0.25 and c_low < collect:
                    dset.append(obs_low)
                    #dset.append(obs_low)
                    c_low += 1
            else:
                obs = obs.detach().cpu().numpy()

                if c_high >= collect and c_low >= collect and c_mid >= collect:
                    break
                elif 0.4 <= pred <= 0.6 and c_mid < collect:
                    dset.append(obs)
                    c_mid += 1
                elif pred >= 0.7 and c_high < collect:
                    dset.append(obs)
                    c_high += 1
                elif pred <= 0.25 and c_low < collect:
                    dset.append(obs)
                    c_low += 1

    #low_val = np.array(dset)
    
    del data # without this line, error gets thrown at the end of the program
    return dset