from __future__ import print_function
from __future__ import division

import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils as utils

from model.resnet import *
from model import utils

#print("PyTorch Version: ",torch.__version__)
#print("Torchvision Version: ",torchvision.__version__)
    

########################## Rotation Estimation ##########################
class KeypointModel(nn.Module):
    
    def __init__(self, name="KeypointModel"):
        super(KeypointModel, self).__init__()

        self.name = name

        self.Keypoint_backBone = resnet34(pretrained=True) 
        
        self.Keypoint_linear = nn.Sequential(
                                    nn.Linear(512, 42),                                     
                                    nn.ReLU()                                                   
                                    )                              

    def forward(self, x):
        
        _, _, _, feature_ext = self.Keypoint_backBone(x)                 
        keyPoint_out = self.Keypoint_linear(feature_ext.view(feature_ext.size()[0], -1))         
        
        return keyPoint_out

 
