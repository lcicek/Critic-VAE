### DEVICE ###
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### TRAINING DETAILS ### 
EPOCHS = 15
NUM_CLASSES = 2 # Just one value between 0 and 1
BATCH_SIZE = 256        
MINERL_SAMPLES = 10000  # Minerl images to extract high/low value images from
DATA_SAMPLES = 2048 # Should divide evenly with batch_size
LHV_IMG_COUNT = DATA_SAMPLES // 2 # High/Low-Value Image count. LV-images = HV-images
CRIT_THRESHOLD = 0.7

n_channels = 3      # Input channels (3 for RGB)
n = 12              # Base amount/multiplier for n_channels of Conv2D
z_dim = 32          # Dimension of latent representation
eps = 1e-7

gen_lr = 0.001
reg_lr = 0.001

collect = BATCH_SIZE * 10  # images are logged every "collect"-step
log_count = 10              # amount of images that should be logged
h = 64 # image height (=width)

### PATHS ###
SAVE_PATH = '/homes/lcicek/Desktop/AAE/images' # change 
CRITIC_PATH = '/homes/lcicek/Desktop/AAE/critic-rewidx=1-cepochs=15-datamode=trunk-datasize=99999-shift=12-chfak=1-dropout=0.3.pt'
MINERL_DATA_ROOT_PATH = '/homes/lcicek/anaconda3/envs/aae/lib/python3.9/site-packages/minerl'
EVAL_IMAGES_PATH = '/homes/lcicek/Desktop/AAE/source-images'
PLOT_DATASET_FILE = 'plot_dataset.pt'

SAVE_IMAGES = True  # Save images locally

### Loss ###
LOSS = 'RCD'