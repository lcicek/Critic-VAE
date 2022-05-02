import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np

from parameters import NUM_CLASSES, CRIT_THRESHOLD

#Encoder
class Q_net(nn.Module):  
    def __init__(self, X_dim, N, z_dim):
        super(Q_net, self).__init__()
        self.model = nn.Sequential(
                        nn.Conv2d(X_dim, N, 3, 1, 1),
                        nn.MaxPool2d(2,2),
                        nn.ReLU(),
                        
                        nn.Conv2d(N, 2 * N, 3, 1, 1),
                        nn.BatchNorm2d(2 * N),
                        nn.MaxPool2d(2),
                        nn.ReLU(),
                        
                        nn.Conv2d(2 * N, 4 * N, 3, 1, 1),
                        nn.BatchNorm2d(4 * N),
                        nn.MaxPool2d(2,2),
                        nn.ReLU(),
                        
                        nn.Conv2d(4 * N, z_dim, 3, 1, 1),
                        nn.BatchNorm2d(z_dim),
                        nn.MaxPool2d(2,2),
                        nn.ReLU(),
                    )

        self.class_output = nn.Sequential(
                                nn.LazyLinear(512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.LazyLinear(256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.LazyLinear(128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.LazyLinear(64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.LazyLinear(32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.LazyLinear(NUM_CLASSES),
                                nn.Sigmoid() 
                        )

        self.wc = nn.Parameter(torch.randn((NUM_CLASSES, 10)))
        
    def forward(self, x_in):
        for layer in self.model:
            x_in = layer(x_in)

        x_model = x_in
        conv_out = torch.flatten(x_model, 1)

        class_out = conv_out
        for layer in self.class_output:
            class_out = layer(class_out)
        
        return class_out, x_model

    def get_shape(self, x):
        x = self.model(x)
        shape = x.shape[1::]
        
        return shape

# Decoder
class P_net(nn.Module):  
    def __init__(self, X_dim, N, z_dim, inner_shape):
        super(P_net, self).__init__()

        self.model = nn.Sequential(
                        # nn.Unflatten(1, inner_shape),
                        
                        #nn.Conv2d(z_dim, 8 * N, 3, 1, 1),
                        #nn.ReLU(),
                        #nn.Upsample(scale_factor=2),  

                        nn.Conv2d(z_dim, 4 * N, 3, 1, 1),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2),
                        
                        nn.Conv2d(4 * N, 2 * N, 3, 1, 1),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2),
                        
                        nn.Conv2d(2 * N, N, 3, 1, 1),
                        nn.ReLU(),  
                        nn.Upsample(scale_factor=2),

                        nn.Conv2d(N, N, 3, 1, 1),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2),
                        
                        nn.Conv2d(N, X_dim, 3, 1, 1),
                        nn.Sigmoid()
                    )

        self.lin_model = nn.Sequential(
                            nn.LazyLinear(16),
                            nn.ReLU(),
        
                            nn.LazyLinear(32),
                            nn.ReLU(),

                            nn.LazyLinear(64),
                            nn.ReLU(),
                            
                            nn.LazyLinear(128),
                            nn.ReLU(),
                            
                            nn.LazyLinear(inner_shape[0]*inner_shape[1]*inner_shape[2]),
                            nn.ReLU(),
                        )
    
    def forward(self, x):
        #for layer in self.lin_model:
        #    x = layer(x)

        for layer in self.model:
            x = layer(x)

        return x

# Discriminator
class D_net(nn.Module):  
    def __init__(self, in_dim, N):
        super(D_net, self).__init__()
        self.model = nn.Sequential(
                        nn.Linear(in_dim, N),
                        nn.ReLU(),
                        nn.Linear(N, N),
                        nn.ReLU(),
                        nn.Linear(N, N),
                        nn.ReLU(), 
                        nn.Linear(N, N),
                        nn.ReLU(),
                        nn.Linear(N, 1),
                        # nn.Sigmoid() # commented out because of BCEwithLogits
                    )
    
    def forward(self, x):
        return self.model(x)

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
            return self.forward(X, collect=True)

    def eval_channels(self, X, alter_layer, alter_channel, toff, collect=False): # toff = turn off 
        with torch.no_grad():
            # X = self.preprocess(X)

            if collect:
                return self.forward_ch_collect(X, alter_layer, alter_channel, toff)
            else:
                return self.forward_ch(X, alter_layer, alter_channel, toff)
