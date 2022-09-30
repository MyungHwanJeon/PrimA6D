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

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils as utils

import argparse
import os
import random
import time
import sys
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model import utils, Discriminator, TraModel, KeypointModel, ReconstModel
from model.utils import *
from model.loss import *

from load_dataset import *
from misc.misc import *

import bop_toolkit_lib.pose_error as PE

import dataset.transform as transfrom
import trimesh

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
parser.add_argument('-b', '--batch', required=False,
                    default=50, type=int,
                    help="batch size")   
parser.add_argument('-e', '--epoch', required=False,
                    default=100, type=int,
                    help="epoch")                                        
                                                                 
args = parser.parse_args()

total_epoch = args.epoch

os.makedirs('./checkpoints/' + args.dataset, exist_ok=True)

model_G_weight_path = "./checkpoints/obj_" + args.obj + "_G.pth"
model_T_weight_path = "./checkpoints/obj_" + args.obj + "_T.pth"

best_te = 10000

print("dataset : ", args.dataset)
print("dataset path : ", args.dataset_path)
print("obj : ", args.obj)
print("batch : ", args.batch)
print("total_epoch : ", total_epoch)

def main():    
    
    batch_size = args.batch
    learning_rate = 0.0001    
        
    ## prepare train dataset ##
    train_file_path = os.path.join(args.dataset_path, "train/obj_%d"%int(args.obj))
    train_file_args = os.listdir(train_file_path)
    valid_args = ["synthetic", "pbr"]
    train_file_list = list()
    for arg in train_file_args:
        if arg in valid_args:
            path = os.path.join(train_file_path, arg)
            if os.path.isdir(path):
                train_file_list = [os.path.join(path, x) for x in os.listdir(path)] 
                
    train_dataset = DatasetLoader(file_dir=train_file_list, train=True)
    train_dataset = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers=8)

    ## prepare test dataset ##
    test_file_path = os.path.join(args.dataset_path, "test/obj_%d"%int(args.obj))
    test_file_list = list()
    if os.path.isdir(test_file_path):
        test_file_list = [os.path.join(test_file_path, x) for x in os.listdir(test_file_path)] 
    
    test_dataset = DatasetLoader(file_dir=test_file_list, train=False)
    test_dataset = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False)   
                
    ## prepare model ##                
    model_G = ReconstModel.ReconstModel().to(device)
    model_G = nn.DataParallel(model_G)
    model_G.cuda()                
    #load_weight_all(model_G, pre_trained_path=model_G_weight_path)  
                
    model_T = TraModel.TranslationModel().to(device)
    model_T = nn.DataParallel(model_T)
    model_T.cuda()         
    optimizer_T = torch.optim.Adam(model_T.parameters(), lr=learning_rate,  weight_decay=1e-5)     
    #load_weight_all(model_T, pre_trained_path=model_T_weight_path)                          
           
    writer = SummaryWriter()           
    for epoch in range(0, total_epoch):    
    
        adjust_learning_rate(optimizer_T, epoch, learning_rate, 50)
        adjust_learning_rate(optimizer_T, epoch, learning_rate, 80)        

        train(train_dataset, model_T, optimizer_T, model_G, epoch, writer)        
        is_best = test(test_dataset, model_T, model_G, writer, epoch)        
                        
        save_checkpoint({'epoch': epoch + 1,
                        'state_dict': model_T.state_dict(),
                        'optimizer' : optimizer_T.state_dict()
                        }, is_best, filename=model_T_weight_path)
        
    writer.close()
        
def train(train_dataset, model_T, optimizer_T, model_G, epoch, writer):

    batch_time = AverageMeter('Time', ':6.3f')
    tra_loss_record = AverageMeter('tra', ':.4f') 
           
    progress_loss = ProgressMeter(int(len(train_dataset)),
                                batch_time,
                                tra_loss_record,
                                prefix="train" + ": [{}]".format(epoch))
    
    
    #switch to train mode
    model_T.train()   
    model_G.eval()    
    
    end = time.time()               
    for i, data in enumerate(train_dataset, 0):

        obj_inp, obj_gt, primitive_gt, obj_bb, rot_gt, tra_gt, kp_2d_gt, K, _ = data   
        obj_inp = obj_inp.to(device)
        obj_gt = obj_gt.to(device)
        primitive_gt = primitive_gt.to(device)        
        kp_2d_gt = kp_2d_gt.to(device)
        tra_gt = tra_gt.to(device)
                        
        ######################################                        
        reconst_out, mu, sigma, primitive_out = model_G(obj_inp)
        
        tra_out, tra_input = model_T(reconst_out, obj_bb)
        
        tra_loss = MSE_loss(tra_out, tra_gt[:,2].view(tra_gt.size()[0], -1))        
        loss = tra_loss
        
        optimizer_T.zero_grad()
        loss.backward()
        optimizer_T.step()  
            
        tra_loss_record.update(tra_loss.item(), obj_inp.size(0))      
                
        if i % 100 == 0:
        
            grid = torchvision.utils.make_grid(tra_input, nrow=16)
            writer.add_image('tra_input', grid, global_step=i) 

            progress_loss.print(i)            
 
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time() 
        

def test(test_dataset, model_T, model_G, writer, epoch):

    batch_time = AverageMeter('Time', ':6.3f')
    te_record = AverageMeter('te', ':.4f') 
    progress_loss = ProgressMeter(int(len(test_dataset)),
                                batch_time,
                                te_record,
                                prefix="test: ")
        
    #switch to eval mode
    model_T.eval()
    model_G.eval()   

    global best_te
    
    for i, data in enumerate(test_dataset):
    
        obj_inp, obj_gt, primitive_gt, obj_bb, rot_gt, tra_gt, kp_2d_gt, K, _ = data   
        obj_inp = obj_inp.to(device)
        obj_gt = obj_gt.to(device)
        primitive_gt = primitive_gt.to(device)        
        kp_2d_gt = kp_2d_gt.to(device)
        tra_gt = tra_gt.to(device)
        
        end = time.time()  
        
        reconst_out, mu, sigma, primitive_out = model_G(obj_inp) 
        
        Tz, _ = model_T(reconst_out, obj_bb) 
        
        z_diff = np.abs(Tz[0, 0].data.cpu().numpy() - tra_gt[0, 2].data.cpu().numpy())
        te_record.update(z_diff, 1)

    is_best = False
    if best_te > te_record.avg:        
        best_te = te_record.avg        
        is_best = True
        
    progress_loss.print(len(test_dataset)) 
    
    return is_best


if __name__ == '__main__':
    main()

