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

class ReconstModel(nn.Module):
    
    def __init__(self, n_latent_code=256, name="ReconstModel"):
        super(ReconstModel, self).__init__()

        self.name = name        
        self.n_latent_code = n_latent_code
        
        ############ Encoder ############
        self.encoder = resnext50_32x4d(pretrained=True)                        
                      
        ############ Variational Sampling ############
        self.mu = nn.Sequential(
                        utils.Reshape(-1, 2048),
                        nn.Linear(2048, self.n_latent_code)
                        )                        
        self.sigma = nn.Sequential(
                        utils.Reshape(-1, 2048),
                        nn.Linear(2048, self.n_latent_code),
                        utils.Absolute()
                        )                        

        ############ Object Reconstruction Decoder ############                
        self.ReconstructionDecoder1 = nn.Sequential(
                                nn.Linear(self.n_latent_code, int(1024*4*4)),
                                nn.ReLU(),
                                utils.Reshape(-1, 1024, 4, 4),
                                nn.ConvTranspose2d(1024, 1024, 3, padding=1, stride=2, output_padding=1),
                                nn.ReLU(),
                                )   
        
        self.ReconstructionDecoder2 = nn.Sequential(
                                nn.Conv2d(2048, 1024, 3, padding=1),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024, 1024, 3, padding=1),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024, 1024, 3, padding=1),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                )                                   
                                
        self.ReconstructionDecoder3 = nn.Sequential(
                                nn.ConvTranspose2d(1024, 512, 3, padding=1, stride=2, output_padding=1),  
                                nn.ReLU(),                                                           
                                )   
                                                                
        self.ReconstructionDecoder4 = nn.Sequential(                       
                                nn.Conv2d(1024, 512, 3, padding=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(512, 512, 3, padding=1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(512, 256, 3, padding=1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                )
                                
        self.ReconstructionDecoder5 = nn.Sequential(
                                nn.ConvTranspose2d(256, 256, 3, padding=1, stride=2, output_padding=1),
                                nn.ReLU(),    
                                )
                                
        self.ReconstructionDecoder6 = nn.Sequential( 
                                nn.Conv2d(512, 256, 3, padding=1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),   
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),   
                                nn.Conv2d(256, 128, 3, padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),   
                                )   
                                                             
        self.ReconstructionDecoder7 = nn.Sequential(
                                nn.ConvTranspose2d(128, 128, 3, padding=1, stride=2, output_padding=1),
                                nn.ReLU(),     
                                nn.Conv2d(128, 64, 3, padding=1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),   
                                nn.Conv2d(64, 3, 3, padding=1),
                                nn.BatchNorm2d(3),
                                nn.ReLU(),   
                                )                                
                
        ############ Primitive Decoder ############
        self.PrimitiveDecoder1 = nn.Sequential(
                        nn.Linear(self.n_latent_code, int(1024*4*4)),
                        nn.ReLU(),
                        utils.Reshape(-1, 1024, 4, 4),  
                        nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
                        #nn.BatchNorm2d(512),
                        nn.ReLU(),                                                                          
                        )
        self.PrimitiveDecoder2 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=1024*2, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.ReLU(),  
                        nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
                        #nn.BatchNorm2d(256),
                        nn.ReLU(),                    
                        )                
        self.PrimitiveDecoder3 = nn.Sequential(                    
                        nn.ConvTranspose2d(in_channels=256*2, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.ReLU(),   
                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                        #nn.BatchNorm2d(128),
                        nn.ReLU(),                                    
                        )                    
        self.PrimitiveDecoder4 = nn.Sequential(                    
                        nn.ConvTranspose2d(in_channels=128*2, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        #nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
                        #nn.BatchNorm2d(3),
                        nn.ReLU()                 
                        )     
                    
        
    def forward(self, x):        
        
        ############ Encoder ############
        encoder_layer1_out, encoder_layer2_out, encoder_layer3_out, feature_map = self.encoder(x)     

        ############ Variational Sampling ############
        mu = self.mu(feature_map)
        sigma = self.sigma(feature_map)                                      
                
        ############ Object Reconstruction Decoder ############        
        sampling_code = self.variational_sampling(mu, sigma)
        
        middle_layer_8_up = self.ReconstructionDecoder1(sampling_code)
        middle_layer_8 = self.ReconstructionDecoder2(torch.cat([middle_layer_8_up, encoder_layer3_out], 1))
        middle_layer_16_up = self.ReconstructionDecoder3(middle_layer_8)
        middle_layer_16 = self.ReconstructionDecoder4(torch.cat([middle_layer_16_up, encoder_layer2_out], 1))
        middle_layer_32_up = self.ReconstructionDecoder5(middle_layer_16)
        middle_layer_32 = self.ReconstructionDecoder6(torch.cat([middle_layer_32_up, encoder_layer1_out], 1))
        reconst_out = self.ReconstructionDecoder7(middle_layer_32)                             
             
        ############ Primitive Decoder ############        
        sampling_code = self.variational_sampling(mu, sigma)
    
        primitive_out = self.PrimitiveDecoder1(sampling_code)    
        primitive_out = self.PrimitiveDecoder2(torch.cat([primitive_out, middle_layer_8], 1)) 
        primitive_out = self.PrimitiveDecoder3(torch.cat([primitive_out, middle_layer_16], 1))         
        primitive_out = self.PrimitiveDecoder4(torch.cat([primitive_out, middle_layer_32], 1))                                    

        return reconst_out, mu, sigma, primitive_out
        
    def variational_sampling(self, mu, sigma):

        normal_distribution = torch.distributions.normal.Normal(torch.zeros(mu.shape), torch.ones(sigma.shape))
        normal_sample = normal_distribution.sample().cuda()

        return mu + sigma*normal_sample        
