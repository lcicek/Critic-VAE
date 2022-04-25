import torch
from torch import nn
import torch.nn.functional as F

from parameters import style_dim, NUM_CLASSES

#Encoder
class Q_net(nn.Module):  
    def __init__(self, X_dim, N, z_dim):
        super(Q_net, self).__init__()
        self.model = nn.Sequential(
                        nn.Conv2d(X_dim, N, 3, 1, 2),
                        nn.MaxPool2d(2,2),
                        nn.ReLU(),
                        
                        nn.Conv2d(N, 2 * N, 3, 1, 2),
                        nn.BatchNorm2d(2 * N),
                        nn.MaxPool2d(2,2),
                        nn.ReLU(),
                        
                        nn.Conv2d(2 * N, 4 * N, 3, 1, 2),
                        nn.BatchNorm2d(4 * N),
                        nn.MaxPool2d(2,2),
                        nn.ReLU(),
                        
                        nn.Conv2d(4 * N, 8 * N, 3, 1, 2),
                        nn.BatchNorm2d(8 * N),
                        nn.MaxPool2d(2,2),
                        nn.ReLU(),
                        
                        nn.Conv2d(8 * N, z_dim, 3, 1, 2),
                        nn.BatchNorm2d(z_dim),
                        nn.MaxPool2d(2,2),
                        nn.ReLU(),   
                    )
        self.lin_model = nn.Sequential(
                                nn.LazyLinear(128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                
                                nn.LazyLinear(64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),

                                nn.LazyLinear(32),
                                nn.ReLU(),
                        )
        
        self.style_output = nn.Sequential(
                                nn.LazyLinear(16),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.LazyLinear(style_dim)
                        )
        self.class_output = nn.Sequential(
                                nn.LazyLinear(16),
                                nn.BatchNorm1d(16),
                                nn.ReLU(),
                                nn.LazyLinear(NUM_CLASSES)
                        )

        self.wc = nn.Parameter(torch.randn((NUM_CLASSES, 2)) * 5)       
        
        
    def forward(self, x_in):
        conv_out = torch.flatten(self.model(x_in), 1)
        x = self.lin_model(conv_out)
        style_out = self.style_output(x)
        class_out = self.class_output(x)
        wc_out = torch.mm(F.softmax(class_out, dim=1), self.wc)
        combined_out = torch.cat([wc_out, style_out], dim=1)  # Concat instead of addition
        return class_out, style_out, combined_out
        
    def get_shape(self, x):
        x = self.model(x)
        s1 = x.shape[1::]
        x = torch.flatten(x, 1)
        x = self.lin_model(x)
        x = self.style_output(x)
        s2 = x.shape[1::] # Exclude batch size
        return s1, s2  

# Decoder
class P_net(nn.Module):  
    def __init__(self, X_dim, N, z_dim, inner_shape):
        super(P_net, self).__init__()
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
        return self.model(self.lin_model(x))

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
