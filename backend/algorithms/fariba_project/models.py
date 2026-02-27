 

# %% [markdown]
# # Import mudoles

# %%
 
# general
import pickle
import time
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.optim as optim
import shutil
import os
import math
import torch.nn.functional as F
from collections import Counter
import numpy as np
import random
# compressai 
from compressai.entropy_models import EntropyBottleneck
from compressai.models.google import JointAutoregressiveHierarchicalPriors
from compressai.layers import GDN
from compressai.models.base import (
    SCALES_LEVELS,
    SCALES_MAX,
    SCALES_MIN,
    CompressionModel,
    get_scale_table,
)
from compressai.models.utils import conv, deconv
__all__ = [
    "CompressionModel",
    "FactorizedPrior",
    "FactorizedPriorReLU",
    "ScaleHyperprior",
    "MeanScaleHyperprior",
    "JointAutoregressiveHierarchicalPriors",
    "get_scale_table",
    "SCALES_MIN",
    "SCALES_MAX",
    "SCALES_LEVELS",
]


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# %%
 
# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
      
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
      
      
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)),inplace=False)
        x = self.bn2(self.conv2(x))
        x = x  +  shortcut
        torch.autograd.set_detect_anomaly(True)
        return F.relu(x,inplace=False)

# %%
class Compression(CompressionModel):
    def __init__(self, N, M , **kwargs):
        super().__init__(**kwargs)
        
        self.encoder = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )
        
        self.decoder = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )
        
        self.entropy_bottleneck = EntropyBottleneck(M)

        self.N = N
        self.M = M
 
        
    def forward(self, x):
      	
        y = self.encoder(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decoder(y_hat)
        torch.autograd.set_detect_anomaly(True)
        
        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "likelihoods": {
                "y": y_likelihoods,
            }
        }

# %%
class Classification(nn.Module):
    def __init__(self,num_classes):
        super().__init__()


        self.classifier = nn.Sequential(
            nn.Conv2d(192,128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size=2),  # Downsampling to reduce spatial size
            
            ResidualBlock(128, 128),  # One ResidualBlock at this level
            nn.MaxPool2d(kernel_size=2),  # Another downsampling step
            
            ResidualBlock(128, 256),  # Move to a higher channel size
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),  # Final downsampling
            
 
            
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling (GAP)
            nn.Flatten(),  # Resulting feature vector is (Batch Size x Channels)
            
            nn.Linear(256, num_classes),  # Fully connected classification layer
            nn.Dropout(0.5,inplace=False)
        )
        self.num_classes = num_classes
    def forward(self, y_hat):
      	
       
        z = self.classifier(y_hat)
        torch.autograd.set_detect_anomaly(True)
    
        
        return {
            "class_output": z
        }

# %%
 