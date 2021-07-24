import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, dropout_value = 0.01):
        super(Net, self).__init__()

        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            # Input: 32x32x3 | Output: 32x32x32 | RF: 3x3

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
            # Input: 32x32x32 | Output: 32x32x64 | RF: 5x5
        ) # Input: 32x32x3 | Output: 32x32x64 | RF: 5x5

        # TRANSITION BLOCK 1
        self.transblock1 = nn.Sequential(
            # Pointwise Convolution to reduce number of channels
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1)), # Input: 32x32x64 | Output: 32x32x32 | RF: 5x5
            
            # Depthwise Convolution with stride=2 to reduce the channel size to half
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=2, groups=32, bias=False)
             # Input: 32x32x32 | Output: 16x16x32 | RF: 7x7
            
        ) # Input: 32x32x64 | Output: 16x16x32 | RF: 7x7

        # CONVOLUTION BLOCK 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value), # Input: 16x16x32 | Output: 16x16x32 | RF: 11x11

            #Depthwise Seperable Convolution

            # Depthwise Convolution
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, groups=32, bias=False),
            # Input: 16x16x32 | Output: 16x16x32 | RF: 15x15
            
            # Pointwise Convolution
            nn.Conv2d(32, 64, kernel_size=1, padding=1),
            # Input: 16x16x32 | Output: 18x18x64 | RF: 15x15
            
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value), 
        ) # Input: 16x16x32 | Output: 18x18x64 | RF: 15x15

        # TRANSITION BLOCK 2
        self.transblock2 = nn.Sequential(
            # Pointwise Convolution to reduce number of channels
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1)), # Input: 18x18x64 | Output: 18x18x32 | RF: 15x15
            
            # Depthwise Convolution with stride=2 to reduce the channel size to half
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=2, groups=32, bias=False)
             # Input: 18x18x32 | Output: 9x9x32 | RF: 19x19
        ) # Input: 18x18x64 | Output: 9x9x32 | RF: 19x19

        # CONVOLUTION BLOCK 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value), # Input: 9x9x32 | Output: 7x7x64 | RF: 35x35
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value), # Input: 7x7x64 | Output: 7x7x64 | RF: 43x43

            #Depthwise Seperable Convolution
            
            # Depthwise Convolution
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, groups=64, bias=False),
            # Input: 7x7x64 | Output: 7x7x64 | RF: 51x51
            
            # Pointwise Convolution
            nn.Conv2d(64, 32, kernel_size=1, padding=1),
            # Input: 7x7x64 | Output: 9x9x32 | RF: 51x51
            
            nn.ReLU(),
            nn.BatchNorm2d(32),
        ) # Input: 9x9x32 | Output: 9x9x32 | RF: 51x51

        # TRANSITION BLOCK 3
        self.transblock3 = nn.Sequential(
            # Pointwise Convolution to reduce number of channels
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1)),  # Input: 9x9x32 | Output: 9x9x16 | RF: 51x51
            
            # Depthwise Convolution with stride=2 to reduce the channel size to half
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, stride=2, groups=16, bias=False)
            # Input: 9x9x16 | Output: 5x5x16 | RF: 59x59
        )# Input: 9x9x32 | Output: 5x5x16 | RF: 59x59

        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
        ) # Input: 5x5x16 | Output: 5x5x10 | RF: 75x75

        # OUTPUT BLOCK
        # Average Pooling to obtain 10-output channels of size 1x1
        self.opblock = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)

        ) # Input: 5x5x10 | Output: 1x1x10 | RF: 107x107


    def forward(self, x):
        x = self.convblock1(x)
        x = self.transblock1(x)
        
        x = self.convblock2(x)
        x = self.transblock2(x)
        
        x = self.convblock3(x)
        x = self.transblock3(x)
        
        x = self.convblock4(x)
        x = self.opblock(x)

        x = x.view(-1, 10)
        return x

#net.load_state_dict(torch.load('/content/96K-model.pth'))