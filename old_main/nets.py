import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

NUM_CLASSES = 2

#Encoder
class Q_net(nn.Module):  
    def __init__(self, X_dim, N, z_dim):
        super(Q_net, self).__init__()
        self.model = nn.Sequential(
                        nn.Conv2d(X_dim, N, 3, 1, 2),
                        nn.MaxPool2d(2,2),
                        nn.ReLU(),
                        
                        nn.Conv2d(N, 2 * N, 3, 1, 2),
                        nn.MaxPool2d(2,2),
                        nn.ReLU(),
                        
                        nn.Conv2d(2 * N, 4 * N, 3, 1, 2),
                        nn.MaxPool2d(2,2),
                        nn.ReLU(),
                        
                        nn.Conv2d(4 * N, 8 * N, 3, 1, 2),
                        nn.MaxPool2d(2,2),
                        nn.ReLU(),
                        
                        nn.Conv2d(8 * N, z_dim, 3, 1, 2),
                        nn.MaxPool2d(2,2),
                        nn.ReLU(),   
                    )
        self.lin_model = nn.Sequential(
                                nn.Linear(z_dim*3*3, 128), # 3*3 is image size after model is done
                                nn.ReLU(),
                                
                                nn.Linear(128, 64),
                                nn.ReLU(),

                                nn.Linear(64, 32),
                                nn.ReLU(),
                        )
        
        self.class_output = nn.Sequential(
                                nn.Linear(32, 16),
                                nn.ReLU(),
                                nn.Linear(16, NUM_CLASSES)
                        )

        self.wc = nn.Parameter(torch.randn((NUM_CLASSES, 2)) * 5)       
        
        
    def forward(self, x_in):
        model_output = self.model(x_in)
        conv_out = torch.flatten(model_output, 1)
        x = self.lin_model(conv_out)
        class_out = self.class_output(x)
        # wc_out = torch.mm(F.softmax(class_out, dim=1), self.wc)
        
        return class_out, conv_out
        
    def get_shape(self, x):
        x = self.model(x)
        s1 = x.shape[1::]
        x = torch.flatten(x, 1)
        x = self.lin_model(x)
        s2 = x.shape[1::] # Exclude batch size
        return s1, s2

# Decoder
class P_net(nn.Module):  
    def __init__(self, X_dim, N, z_dim, inner_shape):
        super(P_net, self).__init__()
        self.lin_model = nn.Sequential(
            nn.Linear(NUM_CLASSES, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, inner_shape[0]*inner_shape[1]*inner_shape[2]),
            nn.ReLU(),
        )

        self.model = nn.Sequential(
            nn.Unflatten(1, inner_shape),
            nn.Conv2d(z_dim, 8 * N, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8 * N, 4 * N, 3, 1, 0),
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
            nn.ReLU(),
            nn.ReflectionPad2d([1,0,1,0]),
            nn.Conv2d(X_dim, X_dim, 2, 1, 0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # lin_x = self.lin_model(x)
        for layer in self.model:
            x = layer(x)

        return x

# Discriminator
class D_net(nn.Module):  
    def __init__(self, N, in_dim):
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
            nn.Linear(N, 1)
        )
    
    def forward(self, x):
        return self.model(x)

class NewCritic(nn.Module):
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
            X = self.preprocess(X)
            return self.forward(X, collect=True)

    def eval_channels(self, X, alter_layer, alter_channel, toff, collect=False): # toff = turn off 
        with torch.no_grad():
            X = self.preprocess(X)

            if collect:
                return self.forward_ch_collect(X, alter_layer, alter_channel, toff)
            else:
                return self.forward_ch(X, alter_layer, alter_channel, toff)