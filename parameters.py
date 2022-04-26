### PATHS ###
SAVE_PATH = '/homes/lcicek/Desktop/AAE/svhn_images_class' # change 
CRITIC_PATH = '/homes/lcicek/Desktop/AAE/critic-rewidx=1-cepochs=15-datamode=trunk-datasize=99999-shift=12-chfak=1-dropout=0.3.pt'
MINERL_DATA_ROOT_PATH = '/homes/lcicek/anaconda3/envs/aae/lib/python3.9/site-packages/minerl'


total_step = 10000 # was 500000
collect = 5000 # images are logged every "collect"-step
log_count = 15 # amount of images that should be logged

EPOCHS = 1
DATA_SAMPLES = 10000 # for RAM-Usage. source: https://github.com/KarolisRam/MineRL2021-Research-baselines/blob/main/standalone/Behavioural_cloning.py#L24        
NUM_CLASSES = 10       # Always 10 for MNIST/SVHN
BATCH_SIZE = 256       # Tune as you like
TRAIN = True           # Whether to execute the training loop cell
AUTOCAST = False       # Possible speedup, but not tested as much. Probably unstable.
SAVE_IMAGES = False    # Save images locally
n_channels = 3         # Input channels. 3 for SVHN, 1 for MNIST
eps = 1e-7             # Only really used in euclidean loss.

style_dim = 10  # Size of style vector. BOTTLENECK OF THE AE-MODEL. 2 is possible for MNIST.
n = 24          # Base amount/multiplier for n_channels of Conv2D
z_dim = 256     # n_channels in last layer before flattening Conv2D output
sample_dim = 12

crit_threshold = 0.75