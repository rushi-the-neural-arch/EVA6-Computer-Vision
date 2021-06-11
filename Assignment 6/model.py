from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dropout_value = 0.05
class Net3_1(nn.Module):
    def __init__(self,normalization_technique):
        super(Net3_1, self).__init__()
        # Input Block
        self.normalization_technique = normalization_technique
        self.convblock1_gn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4, 8),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2_gn = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(8, 16),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.pool1_gn = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4_gn = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock5_gn = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(8, 16),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock6_gn = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(10, 20),
            nn.Dropout(dropout_value)
        ) # output_size = 8

        # 
        self.convblock7_gn = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.GroupNorm(2, 8),
            nn.Dropout(dropout_value)
        ) # output_size = 8

        # OUTPUT BLOCK
        self.convblock8_gn = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 6

        self.gap_gn = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.fc1_gn = nn.Linear(16, 10)
        ##########################################  BN  ##########################################

        self.convblock1_bn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2_bn = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.pool1_bn = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4_bn = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock5_bn = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock6_bn = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(dropout_value)
        ) # output_size = 8

        # 
        self.convblock7_bn = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 8

        # OUTPUT BLOCK
        self.convblock8_bn = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 6

        self.gap_bn = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.fc1_bn = nn.Linear(16, 10)

        ##########################################  LN  ##########################################

        self.convblock1_ln = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm([8,26,26]),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2_ln = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm([16,24,24]),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.pool1_ln = nn.MaxPool2d(2, 2) # output_size = 12
        self.convblock4_ln = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock5_ln = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm([16,10,10]),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock6_ln = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm([20,8,8]),
            nn.Dropout(dropout_value)
        ) # output_size = 8

        # 
        self.convblock7_ln = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.LayerNorm([8,8,8]),
            nn.Dropout(dropout_value)
        ) # output_size = 8

        # OUTPUT BLOCK
        self.convblock8_ln = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 6

        self.gap_ln = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.fc1_ln = nn.Linear(16, 10)
        print("Normalization Technique: ",normalization_technique)

    def forward(self, x):
      if self.normalization_technique == "BN":
        x = self.convblock1_bn(x)
        x = self.convblock2_bn(x)
        #x = self.convblock3(x)
        x = self.pool1_bn(x)
        x = self.convblock4_bn(x)
        x = self.convblock5_bn(x)
        x = self.convblock6_bn(x)
        x = self.convblock7_bn(x)
        x = self.convblock8_bn(x)
        x = self.gap_bn(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1_bn(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

      elif self.normalization_technique == "GN":
        x = self.convblock1_gn(x)
        x = self.convblock2_gn(x)
        #x = self.convblock3(x)
        x = self.pool1_gn(x)
        x = self.convblock4_gn(x)
        x = self.convblock5_gn(x)
        x = self.convblock6_gn(x)
        x = self.convblock7_gn(x)
        x = self.convblock8_gn(x)
        x = self.gap_gn(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1_gn(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

      elif self.normalization_technique == "LN":
        x = self.convblock1_ln(x)
        x = self.convblock2_ln(x)
        #x = self.convblock3(x)
        x = self.pool1_ln(x)
        x = self.convblock4_ln(x)
        x = self.convblock5_ln(x)
        x = self.convblock6_ln(x)
        x = self.convblock7_ln(x)
        x = self.convblock8_ln(x)
        x = self.gap_ln(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1_ln(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

