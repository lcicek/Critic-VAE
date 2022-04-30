### TRAINING DETAILS ### 

EPOCHS = 15
NUM_CLASSES = 2 # High/Low Critic Value
BATCH_SIZE = 256        
DATA_SAMPLES = 75000  # Minerl images (per epoch)
CRIT_THRESHOLD = 0.7

n_channels = 3      # Input channels (3 for RGB)
n = 12              # Base amount/multiplier for n_channels of Conv2D
z_dim = 64          # Dimension of latent representation

gen_lr = 0.001
reg_lr = 0.001

collect = BATCH_SIZE * 10  # images are logged every "collect"-step
log_count = 5              # amount of images that should be logged
h = 64 # image height (=width)

### PATHS ###
SAVE_PATH = '/homes/lcicek/Desktop/AAE/images' # change 
CRITIC_PATH = '/homes/lcicek/Desktop/AAE/critic-rewidx=1-cepochs=15-datamode=trunk-datasize=99999-shift=12-chfak=1-dropout=0.3.pt'
MINERL_DATA_ROOT_PATH = '/homes/lcicek/anaconda3/envs/aae/lib/python3.9/site-packages/minerl'
EVAL_IMAGES_PATH = '/homes/lcicek/Desktop/AAE/source-images'

SAVE_IMAGES = False  # Save images locally

### Loss ###
LOSS = 'RC'