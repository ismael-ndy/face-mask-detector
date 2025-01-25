import torch.nn as nn
import torch.nn.functional as F

class MaskDetectorCNN(nn.Module):
    def __init__(self):
        super(MaskDetectorCNN, self).__init__()
        ## Define the convolutional layers of the model: two convolutional layers with 32 and 64 filters respectively.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        
        ## Define the fully connected layers of the model: two fully connected layers with 128 and 2 units respectively.
        ## ... TODO
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        
        ## Flatten the output of the convolutional layers
        x = x.view(x.size(0), -1)
        
        ## ... TODO