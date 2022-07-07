import numpy as np
import torch
from torch import nn, Tensor

class Critic(nn.Module):
    def __init__(self, width=64, dims=[8, 8, 8, 16], bottleneck=32, colorchs=3,
        chfak=1, activation=nn.ReLU, pool='max', dropout=0.5,
    ):
        super().__init__()
        self.width = width
        stride = 1 if pool == 'max' else 2
        dims = np.array(dims) * chfak
        pool = nn.MaxPool2d(2) if pool == 'max' else nn.Identity()
        self.pool = pool
        features = [
            nn.Conv2d(colorchs, dims[0], 3, stride, 1),
            activation(),
            pool,
            nn.Conv2d(dims[0], dims[1], 3, stride, 1),
            activation(),
            pool,
            nn.Conv2d(dims[1], dims[2], 3, stride, 1),
            activation(),
            pool,
            nn.Dropout(dropout),
            nn.Conv2d(dims[2], dims[3], 3, stride, 1),
            activation(),
            pool,
            nn.Dropout(dropout),
            nn.Conv2d(dims[3], bottleneck * chfak, 4),
            activation(),
        ]
        self.features = nn.Sequential(*features)

        self.crit = nn.Sequential(
            nn.Flatten(),
            nn.Linear(chfak * bottleneck, chfak * bottleneck),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(chfak * bottleneck, 1),
            nn.Sigmoid(),
        )

    def forward(self, X, collect=False):
        embeds = []
        # print(list(self.features))
        for layer in list(self.features):
            X = layer(X)
            if collect and isinstance(layer, type(self.pool)):
                embeds.append(X)
        if collect:
            embeds.append(X)
        # print('last embed', X.shape)
        pred = self.crit(X)

        if collect:
            return pred, embeds
        else:
            return pred

    # toff = turn off; if false then multiply values by coeff. instead of setting to zero
    def forward_ch(self, X, alter_layer, alter_ch, toff):       
        count = 1 # layer 1

        for layer in list(self.features):
            X = layer(X)

            if isinstance(layer, nn.Conv2d):
                if alter_layer == count:
                    size = 2 ** (7 - alter_layer) if alter_layer != 5 else 1 # L1 => 64 = 2^6...; L5 is an exception
                    
                    if toff:
                        X[0][alter_ch] = torch.zeros(size, size) # set channel to 0
                    else:
                        torch.mul(X[0][alter_ch], 5) # !! might have to change it to other number than 5

                count = count + 1

        pred = self.crit(X)

        return pred

    def forward_ch_collect(self, X, alter_layer, alter_ch, toff):
        count = 1 # layer 1
        embeds = []
        finished = False

        for layer in list(self.features):
            if finished:
                break
            
            X = layer(X)

            if count < 4 and isinstance(layer, nn.Conv2d):
                if alter_layer == count:
                    size = 2 ** (7 - alter_layer) if alter_layer != 5 else 1 # L1 => 64 = 2^6...; L5 is an exception
                        
                    if toff:
                        X[0][alter_ch] = torch.zeros(size, size) # set channel to 0
                    else:
                        torch.mul(X[0][alter_ch], 5) # !! might have to change it to other number than 5

                count = count + 1

            if isinstance(layer, nn.ReLU):
                embeds.append(X)

                if count == 4: # if count is 4, then 3 conv layers have been processed already
                    finished = True # so finished is true, since we just look at L1,2,3

            #if isinstance(layer, type(self.pool)):
             #   embeds.append(X)

              #  if count == 4:
               #     finished = True

        return embeds

    def preprocess(self, X: Tensor):
        # X = X.T.unsqueeze(0)
        return (X / 255.0).permute(0, 3, 1, 2).float()
        # return (X/255.0).float()

    def evaluate(self, X): # was called eval_intermediate
        with torch.no_grad():
            # X = self.preprocess(X)
            return self.forward(X, collect=False)

    def eval_channels(self, X, alter_layer, alter_channel, toff, collect=False): # toff = turn off 
        with torch.no_grad():
            # X = self.preprocess(X)

            if collect:
                return self.forward_ch_collect(X, alter_layer, alter_channel, toff)
            else:
                return self.forward_ch(X, alter_layer, alter_channel, toff)
