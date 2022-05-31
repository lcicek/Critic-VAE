### DEVICE ###
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### TRAINING DETAILS ### 
EPOCHS = 15
BATCH_SIZE = 256

NUM_CLASSES = 2 # Just one value between 0 and 1

MINERL_SAMPLES = 200000  # Minerl images to extract high/low value images from
DATA_SAMPLES = 76800 # Should divide evenly with batch_size
LHV_IMG_COUNT = DATA_SAMPLES // 2 # High/Low-Value Image count. LV-images = HV-images
CRIT_THRESHOLD = 0.7

n_channels = 3      # Input channels (3 for RGB)
n = 12              # Base amount/multiplier for n_channels of Conv2D
z_dim = 64          # Dimension of latent representation

reg_lr = 0.0001
gen_lr = 0.001
disc_lr = 0.05

h = 16 # image height (=width)
bottleneck = [64, 1, 1]

log_n = BATCH_SIZE * 30  # data is logged every "log_n"-step

### PATHS ###
SAVE_PATH = '/homes/lcicek/Desktop/AAE/images' # change 
CRITIC_PATH = '/homes/lcicek/Desktop/AAE/saved-networks/critic-rewidx=1-cepochs=15-datamode=trunk-datasize=99999-shift=12-chfak=1-dropout=0.3.pt'
MINERL_DATA_ROOT_PATH = '/homes/lcicek/anaconda3/envs/aae/lib/python3.9/site-packages/minerl'
EVAL_IMAGES_PATH = '/homes/lcicek/Desktop/AAE/source-images'
PLOT_DATASET_PATH = '/homes/lcicek/Desktop/AAE/plot_dataset.pt'
Q_PATH = '/homes/lcicek/Desktop/AAE/saved-networks/Q_encoder_weights.pt'
P_PATH = '/homes/lcicek/Desktop/AAE/saved-networks/P_decoder_weights.pt'
D_PATH = '/homes/lcicek/Desktop/AAE/saved-networks/D_discriminator_weights.pt'

# log_count = 10 # amount of images that should be logged