import torch
from torch import nn, Tensor
from torch.autograd import Variable

threshold = 0.75
eps = 1e-7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gauss_sample():
    real_gauss = torch.rand((1, 1), device=device).item() # (1, 1) because batch_size = 1
    label = 1 if real_gauss >= threshold else 0
    
    return Tensor([[label]]).to(device)

# returns label for single prediction (i.e. when batch_size = 1)
def pred_to_label(pred_list):
    return Tensor([1]).to(device) if pred_list[0].item() >= threshold else Tensor([0]).to(device)

def to_np(x):
    x = x.detach().cpu().numpy() # was x.data before
    return x

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x) 

# Yielded good results but not perfect.
def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            torch.nn.init.normal_(m.bias, 0, 1.414)

def master_params(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            yield p

# Euclidean loss between cluster heads
def euclid_loss(wc):
    dists = torch.cdist(wc.float(), wc.float())
    dists = 1 / (dists + eps)
    dists[dists < 1/4] = 0 # cutoff at distance 4
    dists[dists == 1/eps] = 0 # ignore distances between same cluster-heads
    dists = torch.mean(dists)
    return dists
