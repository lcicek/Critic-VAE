### BEFORE USE: tune parameters in parameters.py, so that DATA_SAMPLES is low 
### torch.save kills program if DATA_SAMPLES is too high ### 

import torch
import os
import minerl
from nets import Critic
from parameters import CRITIC_PATH, MINERL_DATA_ROOT_PATH, PLOT_DATASET_FILE
from utility import load_minerl_data, prepare_data

device = torch.device('cpu')

critic = Critic().to(device)
critic.load_state_dict(torch.load(CRITIC_PATH, map_location=device))

### Initialize mineRL dataset ###
os.environ['MINERL_DATA_ROOT'] = MINERL_DATA_ROOT_PATH
data = minerl.data.make('MineRLTreechop-v0', num_workers=1)
all_obs = load_minerl_data(data, shuffle=False)
del data

dset = prepare_data(all_obs, critic, shuffle=False)
torch.save(dset, PLOT_DATASET_FILE)