total_step = 5000 # was 500000
collect = 5000 # images are logged every "collect"-step
log_count = 15 # amount of images that should be logged

NUM_CLASSES = 10  # Always 10 for MNIST/SVHN
BATCH_SIZE = 200  # Tune as you like
TRAIN = True      # Whether to execute the training loop cell
AUTOCAST = False  # Possible speedup, but not tested as much. Probably unstable.
n_channels = 3    # Input channels. 3 for SVHN, 1 for MNIST
eps = 1e-7        # Only really used in euclidean loss.

style_dim = 10  # Size of style vector. BOTTLENECK OF THE AE-MODEL. 2 is possible for MNIST.
n = 24          # Base amount/multiplier for n_channels of Conv2D
z_dim = 256     # n_channels in last layer before flattening Conv2D output