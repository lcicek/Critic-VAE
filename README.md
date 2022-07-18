# Critic-Variational Autoencoder
![](https://github.com/lcicek/Critic-VAE/blob/master/video-showcase.gif)

## Setup:
##### 1. Clone this repository.
##### 2. Create a conda environment with the necessary packages:
###### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<mark>conda create --name ENV_NAME --file requirements.txt -c conda-forge -c pytorch</mark>
##### 3. Activate the conda environment:
###### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<mark>conda activate ENV_NAME</mark>

## How to run this model:
##### 1. To train the model (results can be found in <mark>'saved-networks/'</mark>), run:
###### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<mark>python vae.py -train</mark>
##### 2. To evaluate the model on the source-images (results can be found in <mark>'images/'</mark>), run:
###### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<mark>python vae.py</mark>
##### 3. To create the mask-video shown above (results can be found in <mark>'videos/'</mark>), run:
###### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<mark>python vae.py -video</mark>
##### _The original minerl-episode is minerl-episode/X.npy and the ground-truth masks are minerl-episode/Y.npy, which are np.arrays containing the 1200 episode images._
##### _Apart from the video, additional information is saved in a bin_info.txt file that is created._

## Other functions of the model:
##### 1. To create a dataset of the reconstructions of the Critic-VAE, run:
###### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<mark>python vae.py -dataset</mark>
##### _The dataset is saved as recon-dataset.pickle and is quite large (~5GB)._
##### _This reconstruction-dataset was unsuccessful so far. To improve it, changes have to be made to the load_minerl_data() function in vae_utility.py._
##### 2. To see the differences of the injected critic-values (results can be found in <mark>'inject/'</mark>), run:
###### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<mark>python vae.py -inject</mark>
##### 3. To test out different thresholds for the thresholded-masks in the video, run:
###### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<mark>python vae.py -video -thresh</mark>
##### _The Intersection over Union (IoU) values of the masks will be printed in the terminal._
##### 4. To train a second Critic-VAE on the reconstruction-dataset (recon-dataset.pickle), run:
###### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<mark>python vae.py -second</mark>
###### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To evaluate it (like above): <mark>python vae.py -evalsecond</mark>
##### _The second Critic-VAE idea was unsuccessful so far as well. Code might be unstable but should be easily fixed with some small changes._
