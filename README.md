# Critic-Variational Autoencoder
![](https://github.com/lcicek/Critic-VAE/blob/master/video-showcase.gif)  
<br/>
Leverages the critic-model: ![Critic](https://github.com/ndrwmlnk/critic-guided-segmentation-of-rewarding-objects-in-first-person-views)

## Setup:
##### 1. Clone this repository.
##### 2. Create a conda environment with the necessary packages:
###### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<mark>conda create --name critvae --file requirements.txt -c conda-forge -c pytorch</mark>
##### 3. Activate the conda environment:
###### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<mark>conda activate critvae</mark>
##### 3. Separately install ![SimpleCRF](https://github.com/HiLab-git/SimpleCRF):
###### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<mark>pip install SimpleCRF</mark>
##### 4. Separately install minerl:
###### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<mark>pip install gym==0.19.0 minerl==0.3.6</mark>
##### 5. Find MINERL_DATA_ROOT.
##### 5.1. MINERL_DATA_ROOT should be located in anaconda-folder: anaconda3/envs/critvae/lib/python3.6/site-packages/minerl
##### 5.2. Find _your_ complete PATH to the MINERL_DATA_ROOT.
##### 5.3. Set MINERL_DATA_ROOT environment variable and download MineRLTreechop-v0 environment:
###### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<mark>MINERL_DATA_ROOT=PATH (replace PATH with yours)</mark>
###### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<mark>export MINERL_DATA_ROOT</mark>
###### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<mark>python3 -m minerl.data.download "MineRLTreechop-v0" (command might differ, see: ![minerl-docs](https://minerl.io/docs/tutorials/data_sampling.html))</mark>
##### 5.4. Replace your PATH with MINERL_DATA_ROOT_PATH in vae_parameters.py.

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
