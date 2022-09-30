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

class PrimA6DNet(nn.Module):
    
    def __init__(self, device=torch.device('cuda')):
        super(PrimA6DNet, self).__init__()

        self.device = device
        self.n_keypoint = 14  

        ############ Encoder ############
        self.primitive_encoder = resnet34(pretrained=True, use_dropout=False).to(device=self.device)         

        ############ Variational Sampling ############
        self.primitive_mu = nn.Sequential(                 
                        utils.Reshape(-1, 512),
                        nn.Linear(512, 128),                        
                        )
                                             

        ############ Primitive Decoder ############
        self.primitive_decoder = nn.Sequential(
                                nn.Linear(128, int(512*4*4), device=self.device),                                
                                nn.LeakyReLU(0.1),
                                
                                utils.Reshape(-1, 512, 4, 4),                                                               
                                nn.ConvTranspose2d(512, 512, 3, padding=1, stride=2, output_padding=1, device=self.device),
                                nn.LeakyReLU(0.1),
                                
                                nn.Conv2d(512, 256, 3, padding=1, device=self.device),
                                nn.LeakyReLU(0.1),       
                                
                                nn.ConvTranspose2d(256, 256, 3, padding=1, stride=2, output_padding=1, device=self.device),  
                                nn.LeakyReLU(0.1),
                                
                                nn.Conv2d(256, 128, 3, padding=1, device=self.device),
                                nn.LeakyReLU(0.1),
                                                              
                                nn.ConvTranspose2d(128, 128, 3, padding=1, stride=2, output_padding=1, device=self.device),
                                nn.LeakyReLU(0.1),
                                
                                nn.Conv2d(128, 64, 3, padding=1, device=self.device),
                                nn.LeakyReLU(0.1),
                                                                            
                                nn.ConvTranspose2d(64, 64, 3, padding=1, stride=2, output_padding=1, device=self.device),
                                nn.LeakyReLU(0.1),
                                
                                nn.Conv2d(64, 32, 3, padding=1, device=self.device),
                                nn.LeakyReLU(0.1),
                                
                                #nn.Conv2d(32, 9, 3, padding=1),
                                #nn.ReLU(),
                                )
                                
        self.primitive_decoder_x = nn.Sequential(
                                nn.Conv2d(32, 3, 3, padding=1, device=self.device),
                                nn.ReLU(),
                                )
                                
        self.primitive_decoder_y = nn.Sequential(
                                nn.Conv2d(32, 3, 3, padding=1, device=self.device),
                                nn.ReLU(),
                                )
                                
        self.primitive_decoder_z = nn.Sequential(
                                nn.Conv2d(32, 3, 3, padding=1, device=self.device),
                                nn.ReLU(),
                                )         
                                
                                
                         
        ######################## keypoint network ########################                           
                                
        self.keypoint_down = nn.Sequential(        
                                nn.Conv2d(9, 3, 3, stride=1, padding=1, device=self.device),
                                nn.ReLU()
                                )                                                                                                  
                                
        self.keypoint_encoder = resnet18(pretrained=True).to(self.device)
        
        
        self.keypoint_decoder_x = nn.Sequential(        
                                    utils.Reshape(-1, 512),
                                    nn.Linear(512, self.n_keypoint*2, device=self.device),
                                    nn.ReLU()
                                    )
                                    
        self.keypoint_decoder_y = nn.Sequential(        
                                    utils.Reshape(-1, 512),
                                    nn.Linear(512, self.n_keypoint*2, device=self.device),
                                    nn.ReLU()
                                    )   
                                    
        self.keypoint_decoder_z = nn.Sequential(        
                                    utils.Reshape(-1, 512),
                                    nn.Linear(512, self.n_keypoint*2, device=self.device),
                                    nn.ReLU()
                                    )                                                                              
        
        self.keypoint_conf = nn.Sequential(        
                                    utils.Reshape(-1, 512),
                                    nn.Linear(512, 3, device=self.device),                                    
                                    )     
                                    
        #self.keypoint_constant = torch.tensor(64., device=self.device)                         
                                                                                                                                     
    def forward(self, x):        
        
        _, _, _, primitive_vec = self.primitive_encoder(x)  
        
        mu = self.primitive_mu(primitive_vec)
                                          
        primitive_middle_out = self.primitive_decoder(mu)
                
        primitive_x_out = self.primitive_decoder_x(primitive_middle_out)
        primitive_y_out = self.primitive_decoder_y(primitive_middle_out)
        primitive_z_out = self.primitive_decoder_z(primitive_middle_out)           
        
        primitive_out = torch.cat((primitive_x_out, primitive_y_out, primitive_z_out), dim=1)
        primitive_reduced = self.keypoint_down(primitive_out)
        
        _, _, _, keypoint_vec = self.keypoint_encoder(primitive_reduced)
        
        keypoint_x_out = self.keypoint_decoder_x(keypoint_vec) * torch.tensor(64., device=self.device)
        keypoint_y_out = self.keypoint_decoder_y(keypoint_vec) * torch.tensor(64., device=self.device)
        keypoint_z_out = self.keypoint_decoder_z(keypoint_vec) * torch.tensor(64., device=self.device)        
        
        keypoint_out = torch.cat((keypoint_x_out, keypoint_y_out, keypoint_z_out), dim=1)
        
        keypoint_uncertainty_out = self.keypoint_conf(keypoint_vec)
                                    
        return primitive_out, keypoint_out, keypoint_uncertainty_out
        
    def variational_sampling(self, mu, sigma):

        normal_distribution = torch.distributions.normal.Normal(torch.zeros(mu.shape), torch.ones(sigma.shape))
        normal_sample = normal_distribution.sample().to(self.device)

        return mu + sigma*normal_sample        
