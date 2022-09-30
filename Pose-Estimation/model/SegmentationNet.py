from __future__ import print_function
from __future__ import division

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

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

class SegmentationNet(nn.Module):
    
    def __init__(self, device=torch.device('cuda')):
        super(SegmentationNet, self).__init__()

        self.device = device
                
        ############ Encoder ############
        self.encoder = resnet18(pretrained=True, use_avgpool=False, use_dropout=False).to(self.device)         

        ############ Primitive Decoder ############
        self.decoder1 = nn.Sequential(    
                                    nn.Conv2d(512, 256, 3, padding=1, device=self.device),
                                    nn.LeakyReLU(0.1),   
                                    nn.ConvTranspose2d(256, 256, 3, padding=1, stride=2, output_padding=1, device=self.device),
                                    nn.LeakyReLU(0.1),     
                                    )
                                    
        self.decoder2 = nn.Sequential(                        
                                    nn.Conv2d(512, 256, 3, padding=1, device=self.device),
                                    nn.LeakyReLU(0.1),   
                                    nn.ConvTranspose2d(256, 128, 3, padding=1, stride=2, output_padding=1, device=self.device),
                                    nn.LeakyReLU(0.1), 
                                    )
                                    
        self.decoder3 = nn.Sequential(                        
                                    nn.Conv2d(256, 128, 3, padding=1, device=self.device),
                                    nn.LeakyReLU(0.1),   
                                    nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2, output_padding=1, device=self.device),
                                    nn.LeakyReLU(0.1), 
                                    )           
                                    
        self.decoder4 = nn.Sequential(                        
                                    nn.Conv2d(128, 64, 3, padding=1, device=self.device),
                                    nn.LeakyReLU(0.1),   
                                    nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2, output_padding=1, device=self.device),
                                    nn.LeakyReLU(0.1), 
                                    nn.Conv2d(32, 1+1, 3, padding=1, device=self.device),   #bkg + 1 class
                                    nn.ReLU(), 
                                    )        
                                    
        self.avg_pool = nn.AdaptiveAvgPool3d((512, 1, 1))                                                                                           
                
    def forward(self, x):        
        
        l1, l2, l3, l4 = self.encoder(x)  

        out = self.decoder1(l4)
        out = self.decoder2(torch.cat((out, l3), 1))
        out = self.decoder3(torch.cat((out, l2), 1))
        out = self.decoder4(torch.cat((out, l1), 1)) 

        return out      
