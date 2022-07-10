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

    def preprocess(self, X: Tensor):
        # X = X.T.unsqueeze(0)
        return (X / 255.0).permute(0, 3, 1, 2).float()
        # return (X/255.0).float()

    def evaluate(self, X): # was called eval_intermediate
        with torch.no_grad():
            # X = self.preprocess(X)
            return self.forward(X, collect=False)
