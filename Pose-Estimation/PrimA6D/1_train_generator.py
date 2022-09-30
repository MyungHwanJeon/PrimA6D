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
                    default=220, type=int,
                    help="epoch")                                        
                                                                 
args = parser.parse_args()

total_epoch = args.epoch

model_G_weight_path = "./checkpoints/obj_" + args.obj + "_G.pth"
model_D_P_weight_path = "./checkpoints/obj_" + args.obj + "_D_P.pth"
model_D_R_weight_path = "./checkpoints/obj_" + args.obj + "_D_R.pth"

best_primitive_loss = 100

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
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=learning_rate,  weight_decay=1e-5)   
    #load_weight_all(model_G, pre_trained_path=model_G_weight_path)   
        
    model_D_R = Discriminator.Discriminator(inp_size=3).to(device) 
    model_D_R = nn.DataParallel(model_D_R)
    model_D_R.cuda()         
    optimizer_D_R = torch.optim.Adam(model_D_R.parameters(), lr=learning_rate,  weight_decay=1e-5)
    #load_weight_all(model_D_R, pre_trained_path=model_D_R_weight_path)
    
    model_D_P = Discriminator.Discriminator(inp_size=3).to(device) 
    model_D_P = nn.DataParallel(model_D_P)
    model_D_P.cuda()         
    optimizer_D_P = torch.optim.Adam(model_D_P.parameters(), lr=learning_rate,  weight_decay=1e-5)
    #load_weight_all(model_D_P, pre_trained_path=model_D_P_weight_path) 
     
    writer = SummaryWriter()                         
    for epoch in range(0, total_epoch):    

        adjust_learning_rate(optimizer_G, epoch, learning_rate, 100)
        adjust_learning_rate(optimizer_D_R, epoch, learning_rate, 100)
        adjust_learning_rate(optimizer_D_P, epoch, learning_rate, 100)

        adjust_learning_rate(optimizer_G, epoch, learning_rate, 180)
        adjust_learning_rate(optimizer_D_R, epoch, learning_rate, 180)
        adjust_learning_rate(optimizer_D_P, epoch, learning_rate, 180)
                      
        train(train_dataset, model_G, optimizer_G, model_D_R, optimizer_D_R, model_D_P, optimizer_D_P, epoch, writer)
        
        is_best = test(test_dataset, model_G, writer, epoch)
        
        save_checkpoint({'epoch': epoch + 1,
                        'state_dict': model_G.state_dict(),
                        'optimizer' : optimizer_G.state_dict()
                        }, False, filename=model_G_weight_path)
                        
        save_checkpoint({'epoch': epoch + 1,
                        'state_dict': model_D_R.state_dict(),
                        'optimizer' : optimizer_D_R.state_dict()
                        }, False, filename=model_D_R_weight_path)
        
        save_checkpoint({'epoch': epoch + 1,
                        'state_dict': model_D_P.state_dict(),
                        'optimizer' : optimizer_D_P.state_dict()
                        }, False, filename=model_D_P_weight_path)
                        
    writer.close()
        

def train(train_dataset, model_G, optimizer_G, model_D_R, optimizer_D_R, model_D_P, optimizer_D_P, epoch, writer):

    batch_time = AverageMeter('Time', ':6.3f')
    kl_div_loss_record = AverageMeter('KL', ':.4f')
    reconst_loss_record = AverageMeter('Reconst', ':.4f')
    primitive_loss_record = AverageMeter('primi', ':.4f')
    G_BCE_loss_record = AverageMeter('G', ':.4f')
    D_R_BCE_loss_record = AverageMeter('D_R', ':.4f')
    D_P_BCE_loss_record = AverageMeter('D_P', ':.4f')
           
    progress_loss = ProgressMeter(int(len(train_dataset)),
                                batch_time,
                                kl_div_loss_record,
                                reconst_loss_record,
                                primitive_loss_record,
                                G_BCE_loss_record,
                                D_R_BCE_loss_record,
                                D_P_BCE_loss_record,
                                prefix="train" + ": [{}]".format(epoch))
        
    #switch to train mode
    model_G.train()   
    model_D_R.train()    
    model_D_P.train()    
    
    end = time.time()       
                            
    for i, data in enumerate(train_dataset, 0):

        obj_inp, obj_gt, primitive_gt, obj_bb, rot_gt, tra_gt, kp_2d_gt, K, _ = data   
        obj_inp = obj_inp.to(device)
        obj_gt = obj_gt.to(device)  
        primitive_gt = primitive_gt.to(device) 

        reconst_out, mu, sigma, primitive_out = model_G(obj_inp)        
        
        if i % 2 is 0:
        
            set_trainable_all(model_G)
            set_trainable_all(model_D_R)
            reset_trainable(model_D_P)  

            ################ train discriminator R ################
            optimizer_D_R.zero_grad()

            pred_fake = model_D_R.forward(reconst_out)
            loss_D_R_fake = BCE_loss(pred_fake, is_real=False)
            
            pred_real = model_D_R.forward(obj_gt)
            loss_D_R_real = BCE_loss(pred_real, is_real=True)

            loss_D_R = (loss_D_R_fake + loss_D_R_real) * 0.5
            loss_D_R.backward(retain_graph=True)   
            optimizer_D_R.step()

            
            ################ train generator ################
            optimizer_G.zero_grad()                
            
            pred_fake = model_D_R.forward(reconst_out)
            loss_G_R_gan = BCE_loss(pred_fake, is_real=True)                      
                
            log_loss = LogLoss()
            log_loss_ = log_loss(reconst_out, obj_gt)
            kl_div_loss = kl_divgence_loss(mu, sigma)
            reconst_loss = Top_K_MSE_loss(reconst_out, obj_gt, k=4)
            primitive_loss = rotation_primitive_loss(primitive_out, primitive_gt)        

            loss_G = loss_G_R_gan + kl_div_loss + reconst_loss + log_loss_ + primitive_loss
            loss_G.backward()
            optimizer_G.step()
            
            
            kl_div_loss_record.update(kl_div_loss.item(), obj_inp.size(0))
            reconst_loss_record.update(reconst_loss.item(), obj_inp.size(0))
            primitive_loss_record.update(primitive_loss.item(), obj_inp.size(0))
            G_BCE_loss_record.update(loss_G.item(), obj_inp.size(0))
            D_R_BCE_loss_record.update(loss_D_R.item(), obj_inp.size(0))
        
        else:
            
            set_trainable_all(model_G)
            reset_trainable(model_D_R)
            set_trainable_all(model_D_P)  
            
            ################ train discriminator P ################
            optimizer_D_P.zero_grad()

            pred_fake = model_D_P.forward(primitive_out)
            loss_D_P_fake = BCE_loss(pred_fake, is_real=False)
            
            pred_real = model_D_P.forward(primitive_gt)
            loss_D_P_real = BCE_loss(pred_real, is_real=True)

            loss_D_P = (loss_D_P_fake + loss_D_P_real) * 0.5
            loss_D_P.backward(retain_graph=True)   
            optimizer_D_P.step()
            
            
            ################ train generator ################
            optimizer_G.zero_grad()
            
            pred_fake = model_D_P.forward(primitive_out)
            loss_G_P_gan = BCE_loss(pred_fake, is_real=True)     
        
            log_loss = LogLoss()
            log_loss_ = log_loss(reconst_out, reconst)
            kl_div_loss = kl_divgence_loss(mu, sigma)
            reconst_loss = Top_K_MSE_loss(reconst_out, reconst, k=4)
            primitive_loss = rotation_primitive_loss(primitive_out, primitive_gt)        

            loss_G = loss_G_P_gan + kl_div_loss + reconst_loss + log_loss_ + primitive_loss
            loss_G.backward()
            optimizer_G.step()
            
            kl_div_loss_record.update(kl_div_loss.item(), obj_inp.size(0))
            reconst_loss_record.update(reconst_loss.item(), obj_inp.size(0))
            primitive_loss_record.update(primitive_loss.item(), obj_inp.size(0))
            G_BCE_loss_record.update(loss_G.item(), obj_inp.size(0))
            D_P_BCE_loss_record.update(loss_D_P.item(), obj_inp.size(0))
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time() 
        
        if i % 100 == 0:

            grid = torchvision.utils.make_grid(obj_inp, nrow=16)
            writer.add_image('train_images', grid, global_step=i)
            grid = torchvision.utils.make_grid(obj_gt, nrow=16)
            writer.add_image('train_reconst_label', grid, global_step=i)
            grid = torchvision.utils.make_grid(primitive_gt, nrow=16)
            writer.add_image('train_primitive_label', grid, global_step=i) 
            grid = torchvision.utils.make_grid(reconst_out, nrow=16)
            writer.add_image('train_reconst_out', grid, global_step=i)            
            grid = torchvision.utils.make_grid(primitive_out, nrow=16)
            writer.add_image('train_primitive_out', grid, global_step=i)    
            
            progress_loss.print(i)    

def test(test_dataset, model_G, writer, epoch):

    primitive_loss_record = AverageMeter('primi', ':.4f')
       
    
    progress_loss = ProgressMeter(int(len(test_dataset)),
                                primitive_loss_record,                                
                                prefix="test" + ": [{}]".format(epoch))
        
    #switch to eval mode
    model_G.eval()   

    global best_primitive_loss

    for i, data in enumerate(test_dataset):
    
        obj_inp, obj_gt, primitive_gt, obj_bb, rot_gt, tra_gt, kp_2d_gt, K, _ = data   
        obj_inp = obj_inp.to(device)
        obj_gt = obj_gt.to(device)  
        primitive_gt = primitive_gt.to(device) 
        
        reconst_out, mu, sigma, primitive_out = model_G(obj_inp) 
        
        grid = torchvision.utils.make_grid(obj_inp, nrow=16)
        writer.add_image('val_images', grid, global_step=i)
        grid = torchvision.utils.make_grid(reconst_out, nrow=16)
        writer.add_image('val_reconst_out', grid, global_step=i)
        grid = torchvision.utils.make_grid(primitive_out, nrow=16)
        writer.add_image('val_primitive_out', grid, global_step=i)   
        
                                 
        primitive_loss = rotation_primitive_loss(primitive_out, primitive_gt)   
        primitive_loss_record.update(primitive_loss.item(), 1)     
        
    is_best = False
    if primitive_loss_record.avg < best_primitive_loss:
        best_primitive_loss = primitive_loss_record.avg
        is_best = True
    print("best_primitive_loss : ", best_primitive_loss)
    
    progress_loss.print(len(test_dataset)) 
        
    return is_best


if __name__ == '__main__':
    main()

