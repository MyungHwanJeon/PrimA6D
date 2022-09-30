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

import numpy as np

from model.resnet import *
from model import utils

#print("PyTorch Version: ",torch.__version__)
#print("Torchvision Version: ",torchvision.__version__)


        
########################## Rotation Estimation ##########################
class TranslationModel(nn.Module):
    
    def __init__(self, name="TranslationModel"):
        super(TranslationModel, self).__init__()

        self.name = name

        self.TranslationBackBone = models.resnet34(pretrained=True)  
        self.TranslationBackBone = nn.Sequential(*list(self.TranslationBackBone.children())[:-1])             
        
        self.TranslationLinear = nn.Sequential(
                    nn.Linear(512, 1),
                    nn.ReLU()                
                    )    
                             
        
    def forward(self, x, obj_bb=None):

        tra_input = torch.zeros(x.size(0), 3, 120, 160).cuda()
        for i in range(x.size(0)):        
                       
            resized_reconst = nn.functional.interpolate(x[i].view(1, x[i].size(0), x[i].size(1), x[i].size(2)), 
                                                        size=(obj_bb[i, 3], obj_bb[i, 2]), 
                                                        mode='bicubic')   
                                                               
            empty_image = torch.zeros(1, 3, 480, 640).cuda()                                                                                
            empty_image[0, :, obj_bb[i,1]:obj_bb[i,1]+obj_bb[i,3], obj_bb[i,0]:obj_bb[i,0]+obj_bb[i,2]] = resized_reconst
                               
            tra_input[i] = nn.functional.interpolate(empty_image, size=(120, 160), mode='bicubic')                        
        
        tra_out = self.TranslationBackBone(tra_input)
        tra_out = tra_out.view(tra_out.size()[0], -1)    
        tra_out = self.TranslationLinear(tra_out)*1000
       
    
        return tra_out, tra_input
        
    def Calcuate_T_using_K(self, T_z, K, obj_center):
        
        bs = T_z.size(0)
        K = torch.from_numpy(K).cuda().float()                        

        T_x = (obj_center[:, 0].view(bs, 1).float()-K[0, 2].repeat(bs).view(bs, 1).float()) * T_z.float() / K[0, 0].repeat(bs).view(bs, 1).float()        
        T_y = (obj_center[:, 1].view(bs, 1).float()-K[1, 2].repeat(bs).view(bs, 1).float()) * T_z.float() / K[1, 1].repeat(bs).view(bs, 1).float()       

        T = torch.zeros([bs, 3]).cuda()
        T[:, 0] = T_x[:, 0]
        T[:, 1] = T_y[:, 0]
        T[:, 2] = T_z[:, 0]

        return T
