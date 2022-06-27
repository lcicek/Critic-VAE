import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

### IMAGE DATA ###
w = 64 # original image width
ch = 3 # image channels

### TRAINING DATA ###
epochs = 8
batch_size = 128
lr = 0.0001
bottleneck = 4096 # 4x4x256; bottleneck of convolutional layers
latent_dim = 64 # fully-connected layer, from 4096 to 64 dim
kld_weight = 0.00025 # note: https://github.com/AntixK/PyTorch-VAE/issues/11 OR https://github.com/AntixK/PyTorch-VAE/issues/35

log_n = batch_size * 30  # data is logged every "log_n"-step
inject_n = 6

### PATHS ###
ENCODER_PATH = 'saved-networks/vae_encoder_weights-regular.pt'
DECODER_PATH = 'saved-networks/vae_decoder_weights-regular.pt'
EVAL_IMAGES_PATH = '/homes/lcicek/Desktop/AAE/source-images'
SAVE_PATH = '/homes/lcicek/Desktop/AAE/second-vae-images'
SAVE_DATASET_PATH = '/homes/lcicek/Desktop/AAE/recon-dataset.pickle'

SECOND_ENCODER_PATH = 'saved-networks/second_vae_encoder_weights.pt'
SECOND_DECODER_PATH = 'saved-networks/second_vae_decoder_weights.pt'

CRITIC_PATH = '/homes/lcicek/Desktop/AAE/saved-networks/critic-rewidx=1-cepochs=15-datamode=trunk-datasize=99999-shift=12-chfak=1-dropout=0.3.pt'
SECOND_CRITIC_PATH = 'recon-critic-net/saves/critic-rewidx=1-cepochs=15-datamode=trunk-datasize=100000-shift=12-chfak=1-dropout=0.3.pt'