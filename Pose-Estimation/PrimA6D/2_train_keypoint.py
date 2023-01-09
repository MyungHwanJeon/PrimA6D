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

os.makedirs('./checkpoints', exist_ok=True)   
os.makedirs('./checkpoints/' + args.dataset, exist_ok=True)

model_G_weight_path = "./checkpoints/" + args.dataset + "/obj_" + args.obj + "_G.pth"
model_K_weight_path = "./checkpoints/" + args.dataset + "/obj_" + args.obj + "_K.pth"

best_re = 10000

obj_model_path = '../dataset/3d_model/' + str(args.dataset) + '/model_eval/obj_' +  "%06d" % int(args.obj) + '.ply'
obj_model = trimesh.load(obj_model_path)
obj_model_scale = 1.0
obj_model.vertices *= obj_model_scale
model_diameter = calc_pts_diameter2(obj_model.vertices)

primitive_model_path = '../dataset/3d_model/PRIMITIVE/3axis.ply'
primitive_model = trimesh.load(primitive_model_path) 
primitive_diameter = calc_pts_diameter2(primitive_model.vertices)
primitive_scale = model_diameter*0.5 / primitive_diameter
primitive_model.vertices *= primitive_scale

kp_3d_4d = np.array([
                    [0, 0, 0, 1],
                    [15, 15, 15, 1], [15, 15, -15, 1], [15, -15, 15, 1], [15, -15, -15, 1], [-15, 15, 15, 1], [-15, 15, -15, 1], [-15, -15, 15, 1], [-15, -15, -15, 1],
                    [105, 15, 15, 1], [105, 15, -15, 1], [105, -15, 15, 1], [105, -15, -15, 1],
                    [15, 105, 15, 1], [15, 105, -15, 1], [-15, 105, 15, 1], [-15, 105, -15, 1],
                    [15, 15, 105, 1], [15, -15, 105, 1], [-15, 15, 105, 1], [-15, -15, 105, 1],
                        ], dtype=np.float32)

kp_3d_4d = np.array(kp_3d_4d, dtype=np.float32)
kp_3d_4d[:, 0:3] = kp_3d_4d[:, 0:3] * primitive_scale

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
    valid_args = ["synthetic", "pbr", "real"]
    train_file_list = list()
    for arg in train_file_args:
        if arg in valid_args:
            path = os.path.join(train_file_path, arg)
            if os.path.isdir(path):
                train_file_list = [os.path.join(path, x) for x in os.listdir(path)] 
                
    train_dataset = DatasetLoader(file_dir=train_file_list, train=True, primitive_scale=primitive_scale)
    train_dataset = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers=8)

    ## prepare test dataset ##
    test_file_path = os.path.join(args.dataset_path, "test/obj_%d"%int(args.obj))
    test_file_list = list()
    if os.path.isdir(test_file_path):
        test_file_list = [os.path.join(test_file_path, x) for x in os.listdir(test_file_path)] 
    
    test_dataset = DatasetLoader(file_dir=test_file_list, train=False, primitive_scale=primitive_scale)
    test_dataset = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False)   
    
    ## prepare model ##                
    model_G = ReconstModel.ReconstModel().to(device)
    model_G = nn.DataParallel(model_G)
    model_G.cuda()                
    #load_weight_all(model_G, pre_trained_path=model_G_weight_path)         

    model_K = KeypointModel.KeypointModel().to(device)
    model_K = nn.DataParallel(model_K)
    model_K.cuda()         
    optimizer_K = torch.optim.Adam(model_K.parameters(), lr=learning_rate,  weight_decay=1e-5)
    #load_weight_all(model_K, pre_trained_path=model_K_weight_path)         
                                             
    writer = SummaryWriter()          
    for epoch in range(0, total_epoch):    
        
        adjust_learning_rate(optimizer_K, epoch, learning_rate, 50)
        adjust_learning_rate(optimizer_K, epoch, learning_rate, 80)                           
                    
        train(train_dataset, model_K, optimizer_K, model_G, epoch, writer)    
        is_best = test(test_dataset, model_K, model_G, writer, epoch)        
                        
        save_checkpoint({'epoch': epoch + 1,
                        'state_dict': model_K.state_dict(),
                        'optimizer' : optimizer_K.state_dict()
                        }, is_best, filename=model_K_weight_path)                           
        
    writer.close()
        

def train(train_dataset, model_K, optimizer_K, model_G, epoch, writer):

    batch_time = AverageMeter('Time', ':6.3f')
    kp_loss_record = AverageMeter('kp', ':.4f') 
           
    progress_loss = ProgressMeter(int(len(train_dataset)),
                                batch_time,
                                kp_loss_record,
                                prefix="train" + ": [{}]".format(epoch))
        
    #switch to train mode
    model_K.train()   
    model_G.eval()    
    
    end = time.time()                                    
    for i, data in enumerate(train_dataset, 0):

        obj_inp, obj_gt, primitive_gt, obj_bb, rot_gt, tra_gt, kp_2d_gt, K, _ = data   
        obj_inp = obj_inp.to(device)
        kp_2d_gt = kp_2d_gt.to(device)
                                                                
        reconst_out, mu, sigma, primitive_out = model_G(obj_inp)                        
        kp_2d_out = model_K(primitive_out)
        
        kp_loss = MSE_loss(kp_2d_out, kp_2d_gt.view(kp_2d_gt.size()[0], -1))        
        loss = kp_loss
        
        optimizer_K.zero_grad()
        loss.backward()
        optimizer_K.step()  
            
        kp_loss_record.update(kp_loss.item(), obj_inp.size(0))                     
         
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time() 
        
        if i % 100 == 0:    
            progress_loss.print(i)    
        

def test(test_dataset, model_K, model_G, writer, epoch):

    batch_time = AverageMeter('Time', ':6.3f')
    re_record = AverageMeter('re', ':.4f')
    progress_loss = ProgressMeter(int(len(test_dataset)),
                                batch_time,
                                re_record,
                                prefix="test: ")
        
    #switch to eval mode
    model_K.eval()   
    model_G.eval()   

    global best_re
    
    true_re = 0             
    for i, data in enumerate(test_dataset):
    
        obj_inp, obj_gt, primitive_gt, obj_bb, rot_gt, tra_gt, kp_2d_gt, K, _ = data   
        obj_inp = obj_inp.to(device)
        kp_2d_gt = kp_2d_gt.to(device)
        K = K.data.cpu().numpy().reshape(3, 3)  
        
        end = time.time()  
        
        reconst_out, mu, sigma, primitive_out = model_G(obj_inp)                     
        keypoint_out = model_K(primitive_out) 
        
        keypoint_out_ = keypoint_out.clone()              
        kp_3d_4d_ = kp_3d_4d.reshape((21, 4))        
        
        obj_bb_mat = torch.zeros(obj_bb.shape[0], 2, 3).to(device)
        obj_bb_mat[:, 0, 2] = obj_bb[:, 0]
        obj_bb_mat[:, 1, 2] = obj_bb[:, 1]
        obj_bb_mat[:, 0, 0] = obj_bb[:, 2] / 64.
        obj_bb_mat[:, 1, 1] = obj_bb[:, 3] / 64.
        
        keypoint_out = keypoint_out.view(-1, kp_3d_4d.shape[0], 2)
        keypoint_out = torch.cat((keypoint_out, torch.ones(keypoint_out.shape[0], keypoint_out.shape[1], 1).to(device)), dim=2)
        kp_out = torch.bmm(obj_bb_mat, keypoint_out.permute(0, 2, 1)).permute(0, 2, 1) 
        kp_out = kp_out.data.cpu().numpy().reshape((21, 2))       
                                 
        retval, rvec, tvec = cv2.solvePnP(
                                        objectPoints = np.ascontiguousarray(kp_3d_4d[:, 0:3].reshape((-1,1,3))), 
                                        imagePoints = np.ascontiguousarray(kp_out[:, 0:2].reshape((-1,1,2))), 
                                        cameraMatrix = K, 
                                        distCoeffs = np.zeros((8, 1), dtype=np.float64),
                                        flags=cv2.SOLVEPNP_ITERATIVE
                                        )                                             
        R_est, _ = cv2.Rodrigues(rvec)   
        
        
        pose_gt = {'R':rot_gt.data.cpu().numpy()[0], 't':tra_gt.data.cpu().numpy().reshape((3, -1))}
        pose = {'R':R_est, 't':tvec.reshape((3, -1))}

        re = PE.re(pose['R'], pose_gt['R'])
        
        re_record.update(re, 1)
                
        if re <= 5: 
            true_re += 1    
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        
        
        ################ pridiction ################
        primitive_np = (primitive_gt[0].data.permute(1, 2, 0).cpu().numpy()*255).copy()
        keypoint_np = kp_2d_gt[0].data.cpu().numpy().reshape((-1, 2))
        for iii in range(keypoint_np.shape[0]):
            primitive_np = cv2.circle(primitive_np, (int(keypoint_np[iii, 0]), int(keypoint_np[iii, 1])), 1, (255, 255, 255), -1)
        primitive_tensor = torch.from_numpy(primitive_np).permute(2, 0, 1).view(1, 3, 64, 64) / 255.                   
                                 
        grid = torchvision.utils.make_grid(obj_inp, nrow=1)
        writer.add_image('val_images', grid, global_step=epoch*len(test_dataset) + i)     
        grid = torchvision.utils.make_grid(primitive_tensor, nrow=1)
        writer.add_image('val_primitive_out', grid, global_step=epoch*len(test_dataset) + i) 
        ##############################################
        
    is_best = False                 
    if re_record.avg < best_re:
        best_re = re_record.avg
        is_best = True                  
    
    progress_loss.print(len(test_dataset)) 
    
    print("----------------------------------------------------")
    print(str(re_record), "\t", true_re/len(test_dataset)*100, "%")
        
    return is_best

if __name__ == '__main__':
    main()

