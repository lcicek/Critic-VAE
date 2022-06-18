import torch
from torch import nn

from parameters import NUM_CLASSES, bottleneck

###------- ENCODER --------###
class Q_net(nn.Module):  
    def __init__(self, X_dim, N, z_dim):
        super(Q_net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(X_dim, N, 5, 1, 2),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            
            nn.Conv2d(N, 2 * N, 5, 1, 2),
            nn.BatchNorm2d(2 * N),
            nn.MaxPool2d(2),
            nn.ReLU(),
            
            nn.Conv2d(2 * N, 4 * N, 5, 1, 2),
            nn.BatchNorm2d(4 * N),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            
            nn.Conv2d(4 * N, z_dim, 5, 1, 2),
            nn.BatchNorm2d(z_dim),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
        )

        self.reduce_dim = nn.LazyLinear(NUM_CLASSES)
        self.class_output = nn.Sigmoid()
        
    def forward(self, x_in):
        for layer in self.model:
            x_in = layer(x_in)

        x_model = x_in
        conv_out = torch.flatten(x_model, 1)
        class_out = self.class_output(self.reduce_dim(conv_out))
        
        return class_out, x_model

    def get_plot_output(self, x):
        with torch.no_grad():
            conv_out = torch.flatten(self.model(x), 1)
            class_out = self.class_output(self.reduce_dim(conv_out))

        return class_out #self.reduce_dim(conv_out)


###------- DECODER --------###
class P_net(nn.Module):  
    def __init__(self, X_dim, N, z_dim, inner_shape=bottleneck):
        super(P_net, self).__init__()

        self.model = nn.Sequential(
            # nn.Unflatten(1, inner_shape),
            
            #nn.Conv2d(z_dim, 8 * N, 3, 1, 1),
            #nn.ReLU(),
            #nn.Upsample(scale_factor=2),  

            nn.Conv2d(z_dim, 4 * N, 5, 1, 2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(4 * N, 2 * N, 5, 1, 2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(2 * N, N, 5, 1, 2),
            nn.ReLU(),  
            nn.Upsample(scale_factor=2),

            nn.Conv2d(N, N, 5, 1, 2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(N, X_dim, 5, 1, 2),
            nn.Sigmoid()
        )

        self.lin_model = nn.Sequential(
            nn.LazyLinear(inner_shape[0]*inner_shape[1]*inner_shape[2]),
            nn.ReLU(),
        )
    
    def forward(self, x, plot=False):
        from parameters import z_dim
        if plot:
            for layer in self.lin_model:
                x = layer(x)
            
            unflatten = nn.Unflatten(1, [z_dim, 1, 1])
            x = unflatten(x)

        for layer in self.model:
            x = layer(x)

        return x

###------- DISCRIMINATOR --------###
class D_net(nn.Module):  
    def __init__(self, in_dim, N):
        super(D_net, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, N),
            nn.ReLU(),
            nn.Linear(N, N),
            nn.ReLU(),
            nn.Linear(N, 1),
            nn.Sigmoid() # comment out when using BCEwithLogits
        )
    
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x