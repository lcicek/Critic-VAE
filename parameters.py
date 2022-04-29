### PATHS ###
SAVE_PATH = '/homes/lcicek/Desktop/AAE/images' # change 
CRITIC_PATH = '/homes/lcicek/Desktop/AAE/critic-rewidx=1-cepochs=15-datamode=trunk-datasize=99999-shift=12-chfak=1-dropout=0.3.pt'
MINERL_DATA_ROOT_PATH = '/homes/lcicek/anaconda3/envs/aae/lib/python3.9/site-packages/minerl'

### Loss ###
LOSS = 'RC'

SAVE_IMAGES = True      # Save images locally

EPOCHS = 5
NUM_CLASSES = 2         # High/Low Critic Value
BATCH_SIZE = 256        # Tune as you like
DATA_SAMPLES = 50000    # Minerl images (per epoch)

TRAIN = True            # Whether to execute the training loop cell
n_channels = 3          # Input channels. 3 for SVHN, 1 for MNIST
eps = 1e-7              # Only really used in euclidean loss.

total_step = 5000 # was 500000
collect = BATCH_SIZE * 10 # images are logged every "collect"-step
log_count = 50 # amount of images that should be logged

style_dim = 10  # Size of style vector. BOTTLENECK OF THE AE-MODEL. 2 is possible for MNIST.
n = 24          # Base amount/multiplier for n_channels of Conv2D
z_dim = 256     # n_channels in last layer before flattening Conv2D output
sample_dim = 10

h = 64 # image height=width
crit_threshold = 0.75

gen_lr = 0.001
reg_lr = 0.001