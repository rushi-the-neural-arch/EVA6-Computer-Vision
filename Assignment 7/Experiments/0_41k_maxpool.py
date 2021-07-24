class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()

    #ConvBlocks1                       #Input Size: (3,32,32) 
    self.convblock1 = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (3,32,32) -> (16,32,32) RF: 3 
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.1),

        #DEPTH-WISE SEP - 1
        nn.Conv2d(in_channels=32,out_channels=64,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (16,32,32) -> (32,32,32) RF: 5 
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout2d(0.1),

        #DEPTH-WISE SEP - 2
        nn.Conv2d(in_channels=64,out_channels=64,groups=64,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (16,32,32) -> (32,32,32) RF: 5 
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout2d(0.1),

        #DEPTH-WISE SEP - 3
        nn.Conv2d(in_channels=64,out_channels=64,groups=64,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (16,32,32) -> (32,32,32) RF: 5 
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout2d(0.1),

        #DEPTH-WISE SEP - 4
        nn.Conv2d(in_channels=64,out_channels=32,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (32,32,32) -> (32,32,32) RF: 7
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.1),

        #transition (1,1)
        nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(1,1),bias=False),                                   # Size: (32,32,32) -> (16,32,32) RF: 7
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=(2,2),stride=2),                                                                  # Size: (16,32,32) -> (16,16,16) RF: 14
    )
    
    self.convblock2 = nn.Sequential(
        
        #DEPTH-WISE SEP - 1
        nn.Conv2d(in_channels=16,out_channels=32,groups=16,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (16,16,16) -> (32,16,16) RF: 16
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.1),


        #DEPTH-WISE SEP - 2
        nn.Conv2d(in_channels=32,out_channels=64,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (16,32,32) -> (32,32,32) RF: 5 
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout2d(0.1),


        #DEPTH-WISE SEP - 3
        nn.Conv2d(in_channels=64,out_channels=64,groups=64,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (16,32,32) -> (32,32,32) RF: 5 
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout2d(0.1),


        #DEPTH-WISE SEP - 4
        nn.Conv2d(in_channels=64,out_channels=64,groups=64,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (32,16,16) -> (32,16,16) RF: 18
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout2d(0.1),


      #DEPTH-WISE SEP - 5
        nn.Conv2d(in_channels=64,out_channels=32,groups=32,kernel_size=(3,3),padding=1,bias=False,padding_mode='replicate'), # Size: (32,16,16) -> (64,16,16) RF: 20
        nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(1,1),bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(0.1),

        #transition (1,1)
        nn.Conv2d(in_channels=32,out_channels=16,kernel_size=(1,1),bias=False),                                    # Size: (64,16,16) -> (16,16,16) RF: 20
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=(2,2),stride=2),                                                                  # Size: (16,16,16) -> (16,8,8) RF: 40
    )

    self.convblock3 = nn.Sequential(
        nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),bias=False),                                    # Size: (16,8,8) -> (16,6,6) RF: 42
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Dropout2d(0.1),

        #transition (1,1)
        nn.Conv2d(in_channels=16,out_channels=10,kernel_size=(1,1),bias=False),                                    # Size: (16,6,6) -> (10,6,6) RF: 42
        nn.ReLU()
    )
  
    '''self.gap = nn.Sequential(
      nn.AdaptiveAvgPool2d(1)
    )'''
    self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=10*6*6,out_features=10,bias=False)
    )

  def forward(self,x):

    x = self.convblock1(x)
    x = self.convblock2(x)
    x = self.convblock3(x)
    x = self.fc(x)
    x = x.view(-1,10)
    return x