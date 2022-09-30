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
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch import autograd

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils


import argparse
import os
import random
import time
import sys
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model import utils, SegmentationNet
from model.utils import *
from model.loss import *

from load_dataset import *
from misc.misc import *

import dataset.transform as transfrom

#cudnn.benchmark = True

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device configuration : ", device)

os.makedirs('./checkpoints', exist_ok=True)

parser = argparse.ArgumentParser(
                                description='R6D')
parser.add_argument('-d', '--dataset', required=False,
                    default="YCB",
                    help="LINEMOD or YCB or TLESS")        
parser.add_argument('-p', '--dataset_path', required=False,
                    default="../dataset/processed_dataset/YCB",
                    help="dataset path")                                                                                                                 
parser.add_argument('-o', '--obj', required=True,
                    default=0,
                    help="object number")                                                                            
                                                                 
args = parser.parse_args()

model_S_weight_path = "./trained_weight/obj_" + args.obj + "_S.pth"

def main():    
    
    ## prepare test dataset ##
    test_file_path = os.path.join(args.dataset_path, "test/obj_%d"%int(args.obj))
    test_file_list = list()
    if os.path.isdir(test_file_path):
        test_file_list = [os.path.join(test_file_path, x) for x in os.listdir(test_file_path)] 
    
    test_dataset = DatasetLoader(file_dir=test_file_list, train=False)
    test_dataset = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False)   
    
    ## prepare segmentaion model ##
    model_S = SegmentationNet.SegmentationNet(device=device).to(device)          
    model_S.cuda()        
    load_weight_all(model_S, pre_trained_path=model_S_weight_path)                     
        
    test(test_dataset, model_S)
            
def test(test_dataset, model_S):
        
    model_S.eval() 
    
    end = time.time() 
    for i, data in enumerate(test_dataset):
    
        obj_color, obj_mask = data
        obj_color = obj_color.to(device)
        obj_mask = obj_mask.to(device).long()   

        ################ pridiction ################
        segmentation_out = model_S(obj_color)
        argmax_segmentation_out = torch.argmax(segmentation_out, dim=1)
           
        cv2.imshow("obj_color", obj_color[0].permute(1, 2, 0).data.cpu().numpy())
        cv2.imshow("segmentation", (argmax_segmentation_out.data.cpu().numpy().reshape(480, 640, 1)*255).astype(np.uint8))        
        cv2.waitKey(100)

        
    return False


if __name__ == '__main__':
    main()

