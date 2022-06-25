import torch

GRAYSCALE = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
w = 64 # original image width
ch = 1 if GRAYSCALE else 3 # image channels

epochs = 8
batch_size = 128
lr = 0.0001
bottleneck = 4096 # 4x4x256; bottleneck of convolutional layers
latent_dim = 64 # fully-connected layer, from 4096 to 64 dim
kld_weight = 0.00025 # note: https://github.com/AntixK/PyTorch-VAE/issues/11 OR https://github.com/AntixK/PyTorch-VAE/issues/35

diff_factor = 5
inject_n = 6
log_n = batch_size * 30  # data is logged every "log_n"-step

ENCODER_PATH = 'saved-networks/vae_encoder_weights-regular.pt'
DECODER_PATH = 'saved-networks/vae_decoder_weights-regular.pt'
EVAL_IMAGES_PATH = '/homes/lcicek/Desktop/AAE/source-images'
SAVE_PATH = '/homes/lcicek/Desktop/AAE/images'
SAVE_DATASET_PATH = '/homes/lcicek/Desktop/AAE/recon-dataset.pickle'
