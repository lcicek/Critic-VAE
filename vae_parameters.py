import torch

GRAYSCALE = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
w = 64 # original image width
ch = 1 if GRAYSCALE else 3 # image channels
epochs = 10
latent_dim = 64
bottleneck = 4096 # 4x4x128; bottleneck of convolutional layers
kld_weight = 0.00005 # https://github.com/AntixK/PyTorch-VAE/issues/11 OR https://github.com/AntixK/PyTorch-VAE/issues/35

diff_factor = 9
inject_n = 6

ENCODER_PATH = 'saved-networks/vae_encoder_weights.pt'
DECODER_PATH = 'saved-networks/vae_decoder_weights.pt'
