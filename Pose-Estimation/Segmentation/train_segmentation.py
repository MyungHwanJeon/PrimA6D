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
                    default=10, type=int,
                    help="batch size")   
parser.add_argument('-e', '--epoch', required=False,
                    default=50, type=int,
                    help="epoch")                                    
                                                                 
args = parser.parse_args()

total_epoch = args.epoch

os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./checkpoints/' + args.dataset, exist_ok=True)

model_S_weight_path = "./checkpoints/" + args.dataset + "/obj_" + args.obj + "_S.pth"

#total_epoch = 10     

best_ADD = 10000000

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
    
    ## prepare segmentaion model ##
    model_S = SegmentationNet.SegmentationNet(device=device).to(device)      
    model_S = nn.DataParallel(model_S) 
    model_S.cuda()        
    #load_weight_all(model_S, pre_trained_path=model_S_weight_path)                     
    optimizer_S = torch.optim.Adam(model_S.parameters(), lr=learning_rate,  weight_decay=1e-5)    
   
    writer = SummaryWriter()                  
    for epoch in range(0, total_epoch):    

        adjust_learning_rate(optimizer_S, epoch, learning_rate, 40)
        adjust_learning_rate(optimizer_S, epoch, learning_rate, 10)
                                    
        train(train_dataset, model_S, optimizer_S, epoch, writer)
        
        is_best = test(test_dataset, model_S, writer, epoch)
        
        save_checkpoint({'epoch': epoch + 1,
                        'state_dict': model_S.module.state_dict(),
                        }, is_best, filename=model_S_weight_path)                                                                  
    writer.close()

def train(train_dataset, model_S, optimizer_S, epoch, writer):

    batch_time = AverageMeter('Time', ':6.3f')
    loss_record = AverageMeter('BCE', ':.4f')  
           
    progress_loss = ProgressMeter(int(len(train_dataset)),
                                batch_time,
                                loss_record,                             
                                prefix="train" + ": [{}]".format(epoch))
        
    model_S.train()   
    
    end = time.time()           
    for i, data in enumerate(train_dataset, 0):

        obj_color, obj_mask = data
        obj_color = obj_color.to(device)
        obj_mask = obj_mask.to(device).long()
        
        ################ pridiction ################
        segmentation_out = model_S(obj_color)
        argmax_segmentation_out = torch.argmax(segmentation_out, dim=1)

        loss_S = CE_loss(segmentation_out, obj_mask)    

        optimizer_S.zero_grad()
        loss_S.backward()
        optimizer_S.step()
                
        ################ record loss ################
        loss_record.update(loss_S.item(), obj_color.size(0))                       
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 100 == 0:
        
            grid = torchvision.utils.make_grid(obj_color, nrow=10)
            writer.add_image('train_input', grid, global_step=epoch*len(train_dataset) + i)            
            grid = torchvision.utils.make_grid(argmax_segmentation_out.view(-1, 1, 480, 640), nrow=10)
            writer.add_image('train_segmentation_out', grid, global_step=epoch*len(train_dataset) + i)            
            
            progress_loss.print(i)   
        
            
            
def test(test_dataset, model_S, writer, epoch):

    batch_time = AverageMeter('Time', ':6.3f')
    loss_record = AverageMeter('BCE', ':.4f')  
    progress_loss = ProgressMeter(int(len(test_dataset)),
                                batch_time,
                                loss_record,  
                                prefix="test" + ": [{}]".format(epoch))
        
    model_S.eval() 
    
    end = time.time() 
    for i, data in enumerate(test_dataset):
    
        obj_color, obj_mask = data
        obj_color = obj_color.to(device)
        obj_mask = obj_mask.to(device).long()   

        ################ pridiction ################
        segmentation_out = model_S(obj_color)
        argmax_segmentation_out = torch.argmax(segmentation_out, dim=1)
        
        loss_S = CE_loss(segmentation_out, obj_mask) 
        
        loss_record.update(loss_S.item(), obj_color.size(0))                       
        batch_time.update(time.time() - end)
        end = time.time()

        grid = torchvision.utils.make_grid(obj_color, nrow=1)
        writer.add_image('val_input', grid, global_step=epoch*len(test_dataset) + i)
        grid = torchvision.utils.make_grid(argmax_segmentation_out.view(-1, 1, 480, 640), nrow=1)
        writer.add_image('val_segmentation_out', grid, global_step=epoch*len(test_dataset) + i)
        
    progress_loss.print(len(test_dataset)) 
        
    return False


if __name__ == '__main__':
    main()

