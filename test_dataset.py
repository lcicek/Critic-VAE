import pickle
import numpy as np
from PIL import Image

with open('/homes/lcicek/Desktop/AAE/recon-dataset.pickle', 'rb') as file:
    recon_dset = pickle.load(file)

traj = np.stack(recon_dset[next(iter(recon_dset))])

img0 = Image.fromarray(traj[100], mode='RGB')
img1 = Image.fromarray(traj[200], mode='RGB')
img2 = Image.fromarray(traj[300], mode='RGB')

img0.save('/homes/lcicek/Desktop/AAE/img0.png', format="png")
img0.save('/homes/lcicek/Desktop/AAE/img1.png', format="png")
img0.save('/homes/lcicek/Desktop/AAE/img2.png', format="png")