class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()

    #ConvBlocks1                       #Input Size: (3,32,32) 
    self.convblock1 = nn.Sequential(
        
        # Plain Conv - 1
        nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (3,32,32) -> (32,32,32) RF: 3 
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #DEPTH-WISE SEP - 1
        nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (32,32,32) -> (64,32,32) RF: 5 
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #DEPTH-WISE SEP - 2
        nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (64,32,32) -> (64,32,32) RF: 7 
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #Dilation Conv - 1
        nn.Conv2d(in_channels=32,out_channels=32,dilation=2,kernel_size=(3,3),padding=1,padding_mode='replicate',bias=False),  # Size: (64,32,32) -> (64,30,30) RF: 11
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #DEPTH-WISE SEP - 3
        nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (64,30,30) -> (64,30,30) RF: 13 
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #DEPTH-WISE SEP - 4
        nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (64,30,30) -> (64,30,30) RF: 15
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #Dilation Conv - 2
        nn.Conv2d(in_channels=32,out_channels=32,dilation=2,kernel_size=(3,3),padding=1,padding_mode='replicate',bias=False),  # Size: (64,28,28) -> (32,28,28) RF: 19
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #transition (1,1)
        nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(1,1),bias=False),                                   # Size: (32,28,28) -> (16,28,28) RF: 19
        nn.ReLU(),

    )
    
    self.convblock2 = nn.Sequential(
        
        # Plain Conv - 2
        nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (16,28,28) -> (32,28,28) RF: 21 
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #DEPTH-WISE SEP - 1
        nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (32,28,28) -> (64,28,28) RF: 23
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),


        #DEPTH-WISE SEP - 2
        nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (64,28,28) -> (64,28,28) RF: 25 
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #Dilation Conv - 3
        nn.Conv2d(in_channels=32,out_channels=32,dilation=2,kernel_size=(3,3),padding=1,padding_mode='replicate',bias=False),  # Size: (64,28,28) -> (64,26,26) RF: 29
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),


        #DEPTH-WISE SEP - 3
        nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (64,26,26) -> (64,26,26) RF: 31 
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),


        #DEPTH-WISE SEP - 4
        nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (64,26,26) -> (64,26,26) RF: 33
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #Dilation Conv - 4
        nn.Conv2d(in_channels=32,out_channels=32,dilation=2,kernel_size=(3,3),padding=1,padding_mode='replicate',bias=False),  # Size: (64,28,28) -> (32,24,24) RF: 37
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),


        #transition (1,1)
        nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(1,1),stride=2,bias=False),                                    # Size: (32,24,24) -> (16,12,12) RF: 37
        nn.ReLU(),

    )

    self.convblock3 = nn.Sequential(
        
        # Plain Conv - 2
        nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (16,24,24) -> (32,24,24) RF: 39 
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #DEPTH-WISE SEP - 5
        nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (32,24,24) -> (64,24,24) RF: 41
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),


        #DEPTH-WISE SEP - 6
        nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (64,24,24) -> (64,24,24) RF: 43 
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #Dilation Conv - 5
        nn.Conv2d(in_channels=32,out_channels=32,dilation=2,kernel_size=(3,3),padding=1,padding_mode='replicate',bias=False), # Size: (64,24,24) -> (64,22,22) RF: 47
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #DEPTH-WISE SEP - 7
        nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (64,22,22) -> (64,22,22) RF: 49
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),


        #DEPTH-WISE SEP - 8
        nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (64,22,22) -> (64,22,22) RF: 51 
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #Dilation Conv - 6
        nn.Conv2d(in_channels=32,out_channels=32,dilation=2,kernel_size=(3,3),padding=1,padding_mode='replicate',bias=False), # Size: (64,22,22) -> (32,20,20) RF: 55
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #transition (1,1)
        nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(1,1),bias=False),                                    # Size: (32,20,20) -> (16,20,20) RF: 55
        nn.ReLU(),

    )

    self.convblock4 = nn.Sequential(
        
        # Plain Conv - 2
        nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (16,20,20) -> (32,20,20) RF: 57 
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #DEPTH-WISE SEP - 5
        nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (32,20,20) -> (64,20,20) RF: 59
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),


        #DEPTH-WISE SEP - 6
        nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (64,20,20) -> (64,20,20) RF: 61 
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #Dilation Conv - 5
        nn.Conv2d(in_channels=32,out_channels=32,dilation=2,kernel_size=(3,3),padding=1,padding_mode='replicate',bias=False),  # Size: (64,20,20) -> (64,18,18) RF: 65
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #DEPTH-WISE SEP - 7
        nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (64,18,18) -> (64,18,18) RF: 67
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),


        #DEPTH-WISE SEP - 8
        nn.Conv2d(in_channels=32,out_channels=32,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (64,18,18) -> (64,18,18) RF: 69 
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #Dilation Conv - 6
        nn.Conv2d(in_channels=32,out_channels=32,dilation=2,kernel_size=(3,3),padding=1,padding_mode='replicate',bias=False),  # Size: (64,18,18) -> (32,16,16) RF: 73
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.05),

        #transition (1,1)
        nn.Conv2d(in_channels=32,out_channels=10,kernel_size=(1,1),bias=False),                                    # Size: (32,16,16) -> (16,16,16) RF: 73
        nn.ReLU(),

    )
    '''self.gap = nn.Sequential(
      nn.AdaptiveAvgPool2d(1)
    )'''
    self.fc = nn.Sequential(
        nn.AvgPool2d(kernel_size=(4,4)),
        # nn.Linear(in_features=16,out_features=10,bias=False)
    )

  def forward(self,x):

    x = self.convblock1(x)
    x = self.convblock2(x)
    x = self.convblock3(x)
    x = self.convblock4(x)
    x = self.fc(x)
    x = x.view(-1,10)
    return x

'''
Device: cuda
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
              ReLU-2           [-1, 32, 32, 32]               0
       BatchNorm2d-3           [-1, 32, 32, 32]              64
         Dropout2d-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]             288
            Conv2d-6           [-1, 32, 32, 32]           1,024
              ReLU-7           [-1, 32, 32, 32]               0
       BatchNorm2d-8           [-1, 32, 32, 32]              64
         Dropout2d-9           [-1, 32, 32, 32]               0
           Conv2d-10           [-1, 32, 32, 32]             288
           Conv2d-11           [-1, 32, 32, 32]           1,024
             ReLU-12           [-1, 32, 32, 32]               0
      BatchNorm2d-13           [-1, 32, 32, 32]              64
        Dropout2d-14           [-1, 32, 32, 32]               0
           Conv2d-15           [-1, 32, 30, 30]           9,216
             ReLU-16           [-1, 32, 30, 30]               0
      BatchNorm2d-17           [-1, 32, 30, 30]              64
        Dropout2d-18           [-1, 32, 30, 30]               0
           Conv2d-19           [-1, 32, 30, 30]             288
           Conv2d-20           [-1, 32, 30, 30]           1,024
             ReLU-21           [-1, 32, 30, 30]               0
      BatchNorm2d-22           [-1, 32, 30, 30]              64
        Dropout2d-23           [-1, 32, 30, 30]               0
           Conv2d-24           [-1, 32, 30, 30]             288
           Conv2d-25           [-1, 32, 30, 30]           1,024
             ReLU-26           [-1, 32, 30, 30]               0
      BatchNorm2d-27           [-1, 32, 30, 30]              64
        Dropout2d-28           [-1, 32, 30, 30]               0
           Conv2d-29           [-1, 32, 28, 28]           9,216
             ReLU-30           [-1, 32, 28, 28]               0
      BatchNorm2d-31           [-1, 32, 28, 28]              64
        Dropout2d-32           [-1, 32, 28, 28]               0
           Conv2d-33           [-1, 16, 28, 28]             512
             ReLU-34           [-1, 16, 28, 28]               0
           Conv2d-35           [-1, 32, 28, 28]           4,608
             ReLU-36           [-1, 32, 28, 28]               0
      BatchNorm2d-37           [-1, 32, 28, 28]              64
        Dropout2d-38           [-1, 32, 28, 28]               0
           Conv2d-39           [-1, 32, 28, 28]             288
           Conv2d-40           [-1, 32, 28, 28]           1,024
             ReLU-41           [-1, 32, 28, 28]               0
      BatchNorm2d-42           [-1, 32, 28, 28]              64
        Dropout2d-43           [-1, 32, 28, 28]               0
           Conv2d-44           [-1, 32, 28, 28]             288
           Conv2d-45           [-1, 32, 28, 28]           1,024
             ReLU-46           [-1, 32, 28, 28]               0
      BatchNorm2d-47           [-1, 32, 28, 28]              64
        Dropout2d-48           [-1, 32, 28, 28]               0
           Conv2d-49           [-1, 32, 26, 26]           9,216
             ReLU-50           [-1, 32, 26, 26]               0
      BatchNorm2d-51           [-1, 32, 26, 26]              64
        Dropout2d-52           [-1, 32, 26, 26]               0
           Conv2d-53           [-1, 32, 26, 26]             288
           Conv2d-54           [-1, 32, 26, 26]           1,024
             ReLU-55           [-1, 32, 26, 26]               0
      BatchNorm2d-56           [-1, 32, 26, 26]              64
        Dropout2d-57           [-1, 32, 26, 26]               0
           Conv2d-58           [-1, 32, 26, 26]             288
           Conv2d-59           [-1, 32, 26, 26]           1,024
             ReLU-60           [-1, 32, 26, 26]               0
      BatchNorm2d-61           [-1, 32, 26, 26]              64
        Dropout2d-62           [-1, 32, 26, 26]               0
           Conv2d-63           [-1, 32, 24, 24]           9,216
             ReLU-64           [-1, 32, 24, 24]               0
      BatchNorm2d-65           [-1, 32, 24, 24]              64
        Dropout2d-66           [-1, 32, 24, 24]               0
           Conv2d-67           [-1, 16, 12, 12]             512
             ReLU-68           [-1, 16, 12, 12]               0
           Conv2d-69           [-1, 32, 12, 12]           4,608
             ReLU-70           [-1, 32, 12, 12]               0
      BatchNorm2d-71           [-1, 32, 12, 12]              64
        Dropout2d-72           [-1, 32, 12, 12]               0
           Conv2d-73           [-1, 32, 12, 12]             288
           Conv2d-74           [-1, 32, 12, 12]           1,024
             ReLU-75           [-1, 32, 12, 12]               0
      BatchNorm2d-76           [-1, 32, 12, 12]              64
        Dropout2d-77           [-1, 32, 12, 12]               0
           Conv2d-78           [-1, 32, 12, 12]             288
           Conv2d-79           [-1, 32, 12, 12]           1,024
             ReLU-80           [-1, 32, 12, 12]               0
      BatchNorm2d-81           [-1, 32, 12, 12]              64
        Dropout2d-82           [-1, 32, 12, 12]               0
           Conv2d-83           [-1, 32, 10, 10]           9,216
             ReLU-84           [-1, 32, 10, 10]               0
      BatchNorm2d-85           [-1, 32, 10, 10]              64
        Dropout2d-86           [-1, 32, 10, 10]               0
           Conv2d-87           [-1, 32, 10, 10]             288
           Conv2d-88           [-1, 32, 10, 10]           1,024
             ReLU-89           [-1, 32, 10, 10]               0
      BatchNorm2d-90           [-1, 32, 10, 10]              64
        Dropout2d-91           [-1, 32, 10, 10]               0
           Conv2d-92           [-1, 32, 10, 10]             288
           Conv2d-93           [-1, 32, 10, 10]           1,024
             ReLU-94           [-1, 32, 10, 10]               0
      BatchNorm2d-95           [-1, 32, 10, 10]              64
        Dropout2d-96           [-1, 32, 10, 10]               0
           Conv2d-97             [-1, 32, 8, 8]           9,216
             ReLU-98             [-1, 32, 8, 8]               0
      BatchNorm2d-99             [-1, 32, 8, 8]              64
       Dropout2d-100             [-1, 32, 8, 8]               0
          Conv2d-101             [-1, 16, 8, 8]             512
            ReLU-102             [-1, 16, 8, 8]               0
          Conv2d-103             [-1, 32, 8, 8]           4,608
            ReLU-104             [-1, 32, 8, 8]               0
     BatchNorm2d-105             [-1, 32, 8, 8]              64
       Dropout2d-106             [-1, 32, 8, 8]               0
          Conv2d-107             [-1, 32, 8, 8]             288
          Conv2d-108             [-1, 32, 8, 8]           1,024
            ReLU-109             [-1, 32, 8, 8]               0
     BatchNorm2d-110             [-1, 32, 8, 8]              64
       Dropout2d-111             [-1, 32, 8, 8]               0
          Conv2d-112             [-1, 32, 8, 8]             288
          Conv2d-113             [-1, 32, 8, 8]           1,024
            ReLU-114             [-1, 32, 8, 8]               0
     BatchNorm2d-115             [-1, 32, 8, 8]              64
       Dropout2d-116             [-1, 32, 8, 8]               0
          Conv2d-117             [-1, 32, 6, 6]           9,216
            ReLU-118             [-1, 32, 6, 6]               0
     BatchNorm2d-119             [-1, 32, 6, 6]              64
       Dropout2d-120             [-1, 32, 6, 6]               0
          Conv2d-121             [-1, 32, 6, 6]             288
          Conv2d-122             [-1, 32, 6, 6]           1,024
            ReLU-123             [-1, 32, 6, 6]               0
     BatchNorm2d-124             [-1, 32, 6, 6]              64
       Dropout2d-125             [-1, 32, 6, 6]               0
          Conv2d-126             [-1, 32, 6, 6]             288
          Conv2d-127             [-1, 32, 6, 6]           1,024
            ReLU-128             [-1, 32, 6, 6]               0
     BatchNorm2d-129             [-1, 32, 6, 6]              64
       Dropout2d-130             [-1, 32, 6, 6]               0
          Conv2d-131             [-1, 32, 4, 4]           9,216
            ReLU-132             [-1, 32, 4, 4]               0
     BatchNorm2d-133             [-1, 32, 4, 4]              64
       Dropout2d-134             [-1, 32, 4, 4]               0
          Conv2d-135             [-1, 10, 4, 4]             320
            ReLU-136             [-1, 10, 4, 4]               0
       AvgPool2d-137             [-1, 10, 1, 1]               0
================================================================
Total params: 113,056
Trainable params: 113,056
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 14.39
Params size (MB): 0.43
Estimated Total Size (MB): 14.84
----------------------------------------------------------------
'''