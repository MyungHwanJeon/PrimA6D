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
import json

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model import utils, Discriminator, PrimA6DPPNet
from model.utils import *
from model.loss import *

from load_dataset import *
from misc.misc import *

import bop_toolkit_lib.pose_error as PE
import bop_toolkit_lib.renderer as renderer

import dataset.transform as transfrom
import trimesh

#cudnn.benchmark = True

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device configuration : ", device)

os.makedirs('./checkpoints', exist_ok=True)   
            
parser = argparse.ArgumentParser(description='R6D')
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

os.makedirs('./checkpoints/' + args.dataset, exist_ok=True)

model_P_weight_path = "./checkpoints/" + args.dataset + "/obj_" + args.obj + "_P.pth"
model_D_weight_path = "./checkpoints/" + args.dataset + "/obj_" + args.obj + "_D.pth"

best_add = 10000000
best_adi = 10000000
best_vsd = 10000000

obj_model_path = '../dataset/3d_model/' + str(args.dataset) + '/model_eval/obj_' +  "%06d" % int(args.obj) + '.ply'
obj_model = trimesh.load(obj_model_path)
obj_model_scale = 1.0
obj_model.vertices *= obj_model_scale
model_diameter = calc_pts_diameter2(obj_model.vertices)

primitive_model_path = '../dataset/3d_model/PRIMITIVE/1axis.ply'
primitive_model = trimesh.load(primitive_model_path) 
primitive_diameter = calc_pts_diameter2(primitive_model.vertices)
primitive_scale = model_diameter*0.5 / primitive_diameter
primitive_model.vertices *= primitive_scale

bop_renderer = None
if args.dataset == "TLESS":
    bop_renderer = renderer.create_renderer(720, 540, "vispy", mode='depth')
else:
    bop_renderer = renderer.create_renderer(640, 480, "vispy", mode='depth')
bop_renderer.add_object(int(args.obj), obj_model_path)         
                                                
kp_3d_4d = np.array([                           
                    [[0, 0, 0, 1], [15, 15, 15, 1], [15, 15, -15, 1], [15, -15, 15, 1], [15, -15, -15, 1], [-15, 15, 15, 1], [-15, 15, -15, 1], [-15, -15, 15, 1], [-15, -15, -15, 1],
                    [105, 0, 0, 1], [105, 15, 15, 1], [105, 15, -15, 1], [105, -15, 15, 1], [105, -15, -15, 1]],                  
                      
                    [[0, 0, 0, 1], [15, 15, 15, 1], [15, 15, -15, 1], [15, -15, 15, 1], [15, -15, -15, 1], [-15, 15, 15, 1], [-15, 15, -15, 1], [-15, -15, 15, 1], [-15, -15, -15, 1],
                    [0, 105, 0, 1], [15, 105, 15, 1], [15, 105, -15, 1], [-15, 105, 15, 1], [-15, 105, -15, 1]],                    
                    
                    [[0, 0, 0, 1], [15, 15, 15, 1], [15, 15, -15, 1], [15, -15, 15, 1], [15, -15, -15, 1], [-15, 15, 15, 1], [-15, 15, -15, 1], [-15, -15, 15, 1], [-15, -15, -15, 1],
                    [0, 0, 105, 1], [15, 15, 105, 1], [15, -15, 105, 1], [-15, 15, 105, 1], [-15, -15, 105, 1]],                                        
                        ], dtype=np.float32)   
                                                                                                                                           
kp_3d_4d = np.array(kp_3d_4d, dtype=np.float32)                                              
kp_3d_4d[:, :, 0:3] = kp_3d_4d[:, :, 0:3] * primitive_scale

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
                
    train_dataset = DatasetLoader(file_dir=train_file_list, primitive_scale=primitive_scale, train=True)
    train_dataset = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers=8)

    ## prepare test dataset ##
    test_file_path = os.path.join(args.dataset_path, "test/obj_%d"%int(args.obj))
    test_file_list = list()
    if os.path.isdir(test_file_path):
        test_file_list = [os.path.join(test_file_path, x) for x in os.listdir(test_file_path)] 
    
    test_dataset = DatasetLoader(file_dir=test_file_list, primitive_scale=primitive_scale, train=False)
    test_dataset = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False)    
                
    ## prepare model ##   
    model_P = PrimA6DPPNet.PrimA6DNet(device=device).to(device)
    model_P.cuda()                          
    #load_weight_all(model_P, pre_trained_path=model_P_weight_path)              
    optimizer_P = torch.optim.Adam(model_P.parameters(), lr=learning_rate,  weight_decay=1e-5)  

    model_D = Discriminator.Discriminator(inp_size=9).to(device) 
    model_D.cuda()             
    #load_weight_all(model_D, pre_trained_path=model_D_weight_path)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=learning_rate,  weight_decay=1e-5)

    writer = SummaryWriter()   
    for epoch in range(0, total_epoch):    

        adjust_learning_rate(optimizer_P, epoch, learning_rate, 100)
        adjust_learning_rate(optimizer_D, epoch, learning_rate, 100)  

        adjust_learning_rate(optimizer_P, epoch, learning_rate, 180)
        adjust_learning_rate(optimizer_D, epoch, learning_rate, 180)    
            
        train(train_dataset, model_P, optimizer_P, model_D, optimizer_D, epoch, writer)        
        
        is_best_vsd, is_best_add, is_best_adi = test(test_dataset, model_P, writer, epoch)
        
        save_checkpoint({'epoch': epoch + 1,
                        'state_dict': model_P.state_dict(),
                        }, False, filename=model_P_weight_path)        
                                                                  
        if is_best_add:
            save_checkpoint({'epoch': epoch + 1,
                    'state_dict': model_P.state_dict(),
                    }, False, filename=model_P_weight_path+"_add_best")     
                    
        if is_best_adi:
            save_checkpoint({'epoch': epoch + 1,
                    'state_dict': model_P.state_dict(),
                    }, False, filename=model_P_weight_path+"_adi_best")                                                          

        if is_best_vsd:
            save_checkpoint({'epoch': epoch + 1,
                    'state_dict': model_P.state_dict(),
                    }, False, filename=model_P_weight_path+"_vsd_best")                                           

        save_checkpoint({'epoch': epoch + 1,
                        'state_dict': model_D.state_dict(),
                        }, False, filename=model_D_weight_path)                                                                                   
                     
                        
    writer.close()
        
def train(train_dataset, model_P, optimizer_P, model_D, optimizer_D, epoch, writer):

    batch_time = AverageMeter('Time', ':6.3f')
    primitive_x_loss_record = AverageMeter('pri_x', ':.4f') 
    primitive_y_loss_record = AverageMeter('pri_y', ':.4f') 
    primitive_z_loss_record = AverageMeter('pri_z', ':.4f')
    keypoint_x_loss_record =  AverageMeter('kp_x', ':.4f')
    keypoint_y_loss_record =  AverageMeter('kp_y', ':.4f')
    keypoint_z_loss_record =  AverageMeter('kp_z', ':.4f')
           
    progress_loss = ProgressMeter(int(len(train_dataset)),
                                batch_time,
                                primitive_x_loss_record,
                                primitive_y_loss_record,
                                primitive_z_loss_record,   
                                keypoint_x_loss_record,
                                keypoint_y_loss_record,
                                keypoint_z_loss_record,
                                prefix="train" + ": [{}]".format(epoch))
        
    #switch to train mode
    model_P.train()   
    model_D.train()   
            
    end = time.time()           
    for i, data in enumerate(train_dataset, 0):
        
        obj_inp, obj_gt, primitive_gt, primitive_x_gt, primitive_y_gt, primitive_z_gt, obj_bb, rot_gt, tra_gt, kp_2d_gt, K, obj_depth = data
            
        obj_inp = obj_inp.to(device)  
        primitive_gt = primitive_gt.to(device)
        primitive_x_gt = primitive_x_gt.to(device)
        primitive_y_gt = primitive_y_gt.to(device)
        primitive_z_gt = primitive_z_gt.to(device)
        rot_gt = rot_gt.to(device)
        tra_gt = tra_gt.to(device)
        kp_2d_gt = kp_2d_gt.to(device)   
        K = K.to(device)   
        
        ################ pridiction ################            
        primitive_out, keypoint_out, keypoint_uncertainty_out = model_P(obj_inp)
        
        ################ train discriminator ################
        optimizer_D.zero_grad()

        generation_fake = model_D(primitive_out)
        loss_D_fake = BCE_loss(generation_fake, is_real=False)    

        generation_real = model_D(torch.cat((primitive_x_gt, primitive_y_gt, primitive_z_gt), dim=1))    
        loss_D_real = BCE_loss(generation_real, is_real=True)
        
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward(retain_graph=True)
        optimizer_D.step() 
        
        ################ train generator ################        
        generation_fake = model_D(primitive_out)
        loss_D_gan = BCE_loss(generation_fake, is_real=True)                      

        primitive_x_loss = rotation_primitive_loss(primitive_out[:, :3], primitive_x_gt)
        primitive_y_loss = rotation_primitive_loss(primitive_out[:, 3:6], primitive_y_gt)
        primitive_z_loss = rotation_primitive_loss(primitive_out[:, 6:9], primitive_z_gt)
        
        keypoint_x_loss = keypoint_uncertainty_loss(keypoint_out[:, :28].reshape(-1, 28), kp_2d_gt[:, 0, :].reshape(-1, 28).float().detach(), keypoint_uncertainty_out[:, 0])
        keypoint_y_loss = keypoint_uncertainty_loss(keypoint_out[:, 28:56].reshape(-1, 28), kp_2d_gt[:, 1, :].reshape(-1, 28).float().detach(), keypoint_uncertainty_out[:, 1])
        keypoint_z_loss = keypoint_uncertainty_loss(keypoint_out[:, 56:84].reshape(-1, 28), kp_2d_gt[:, 2, :].reshape(-1, 28).float().detach(), keypoint_uncertainty_out[:, 2])

        loss_P = loss_D_gan + primitive_x_loss + primitive_y_loss + primitive_z_loss + keypoint_x_loss + keypoint_y_loss + keypoint_z_loss
                          
        optimizer_P.zero_grad()
        loss_P.backward()
        optimizer_P.step()        
                
        ################ record loss ################
        primitive_x_loss_record.update(primitive_x_loss.item(), obj_inp.size(0))                      
        primitive_y_loss_record.update(primitive_y_loss.item(), obj_inp.size(0))                      
        primitive_z_loss_record.update(primitive_z_loss.item(), obj_inp.size(0))    
        keypoint_x_loss_record.update(keypoint_x_loss.item(), obj_inp.size(0))                      
        keypoint_y_loss_record.update(keypoint_y_loss.item(), obj_inp.size(0))                      
        keypoint_z_loss_record.update(keypoint_z_loss.item(), obj_inp.size(0))                  

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 100 == 0:
        
            grid = torchvision.utils.make_grid(obj_inp, nrow=10)
            writer.add_image('train_input', grid, global_step=epoch*len(train_dataset) + i)
            
            grid = torchvision.utils.make_grid(primitive_out[:, 0:3, :, :], nrow=10)
            writer.add_image('train_primitive_x_out', grid, global_step=epoch*len(train_dataset) + i)
            grid = torchvision.utils.make_grid(primitive_out[:, 3:6, :, :], nrow=10)
            writer.add_image('train_primitive_y_out', grid, global_step=epoch*len(train_dataset) + i)
            grid = torchvision.utils.make_grid(primitive_out[:, 6:9, :, :], nrow=10)
            writer.add_image('train_primitive_z_out', grid, global_step=epoch*len(train_dataset) + i)
            
            grid = torchvision.utils.make_grid(primitive_x_gt, nrow=10)
            writer.add_image('train_primitive_x_gt', grid, global_step=epoch*len(train_dataset) + i)
            grid = torchvision.utils.make_grid(primitive_y_gt, nrow=10)
            writer.add_image('train_primitive_y_gt', grid, global_step=epoch*len(train_dataset) + i)
            grid = torchvision.utils.make_grid(primitive_z_gt, nrow=10)
            writer.add_image('train_primitive_z_gt', grid, global_step=epoch*len(train_dataset) + i)                        
            
            progress_loss.print(i) 
            
def test(test_dataset, model_P, writer, epoch):

    batch_time = AverageMeter('Time', ':6.3f')
    VSD_record = AverageMeter('VSD', ':.4f')
    ADD_record = AverageMeter('ADD', ':.4f')
    ADI_record = AverageMeter('ADI', ':.4f')    
    PROJ_record = AverageMeter('PROJ', ':.4f')
    RE_record = AverageMeter('RE', ':.4f')
    TE_record = AverageMeter('TE', ':.4f')
    
    progress_loss = ProgressMeter(int(len(test_dataset)),
                                    batch_time,
                                    VSD_record,
                                    ADD_record,
                                    ADI_record,
                                    PROJ_record,
                                    RE_record,
                                    TE_record,
                                    prefix="test" + ": [{}]".format(epoch))
        
    #switch to eval mode
    model_P.eval() 
    
    global best_add
    global best_adi
    global best_vsd

    true_vsd = 0    
    true_add = 0
    true_adi = 0    
    true_proj = 0
    true_re = 0
    true_te = 0 
    
    valid_case = 0
    
    end = time.time()  
    for i, data in enumerate(test_dataset):
    
        obj_inp, obj_gt, primitive_gt, primitive_x_gt, primitive_y_gt, primitive_z_gt, obj_bb, rot_gt, tra_gt, kp_2d_gt, K, obj_depth = data      
        obj_inp = obj_inp.to(device)  
        primitive_gt = primitive_gt.to(device)
        primitive_x_gt = primitive_x_gt.to(device)
        primitive_y_gt = primitive_y_gt.to(device)
        primitive_z_gt = primitive_z_gt.to(device)
        rot_gt = rot_gt.to(device)
        tra_gt = tra_gt.to(device)
        kp_2d_gt = kp_2d_gt.to(device)   
        K = K.data.cpu().numpy().reshape(3, 3)  
        if obj_depth is not None:
            obj_depth = obj_depth.data.cpu().numpy()

        ################ pridiction ################
        primitive_out, keypoint_out, keypoint_uncertainty_out = model_P(obj_inp)
                                             
        keypoint_out_ = keypoint_out.clone()              
        kp_3d_4d_ = kp_3d_4d.reshape((14*3, 4))        
        
        obj_bb_mat = torch.zeros(obj_bb.shape[0], 2, 3).to(device)
        obj_bb_mat[:, 0, 2] = obj_bb[:, 0]
        obj_bb_mat[:, 1, 2] = obj_bb[:, 1]
        obj_bb_mat[:, 0, 0] = obj_bb[:, 2] / 64.
        obj_bb_mat[:, 1, 1] = obj_bb[:, 3] / 64.
        
        keypoint_out = keypoint_out.view(-1, kp_3d_4d_.shape[0], 2)
        keypoint_out = torch.cat((keypoint_out, torch.ones(keypoint_out.shape[0], keypoint_out.shape[1], 1).to(device)), dim=2)
        kp_out = torch.bmm(obj_bb_mat, keypoint_out.permute(0, 2, 1)).permute(0, 2, 1)  
        
        kp_out = kp_out.data.cpu().numpy().reshape((14*3, 2))
        
        threshold = 0.4
        valid_cnt = 0
        valid_axis = np.array([False, False, False], dtype=np.bool)
        
        if torch.exp(keypoint_uncertainty_out[0, 0]) < threshold:
            valid_cnt += 1
            valid_axis[0] = True
        if torch.exp(keypoint_uncertainty_out[0, 1]) < threshold:
            valid_cnt += 1
            valid_axis[1] = True
        if torch.exp(keypoint_uncertainty_out[0, 2]) < threshold:
            valid_cnt += 1
            valid_axis[2] = True
            
        
        if valid_cnt > 0:
        
            kp_3d_pts = np.zeros((9 + valid_cnt * 5, 4), dtype=np.float32)
            kp_2d_pts = np.zeros((9 + valid_cnt * 5, 2), dtype=np.float32)
            
            start_idx = 9                            
            if valid_axis[0]:#is True:         
            
                for iii in range(9):
                    kp_3d_pts[iii] = kp_3d_4d_[iii]
                    kp_2d_pts[iii] += kp_out[iii]
                   
                for iii in range(5):
                    kp_3d_pts[start_idx + iii] = kp_3d_4d_[0 + 9 + iii]
                    kp_2d_pts[start_idx + iii] = kp_out[0 + 9 + iii]
                start_idx += 5    
                    
            if valid_axis[1]:#is True:     
            
                for iii in range(9):
                    kp_3d_pts[iii] = kp_3d_4d_[iii]
                    kp_2d_pts[iii] += kp_out[14 + iii]
                           
                for iii in range(5):
                    kp_3d_pts[start_idx + iii] = kp_3d_4d_[14 + 9 + iii]
                    kp_2d_pts[start_idx + iii] = kp_out[14 + 9 + iii]
                start_idx += 5
                    
            if valid_axis[2]:# is True:        
            
                for iii in range(9):
                    kp_3d_pts[iii] = kp_3d_4d_[iii]
                    kp_2d_pts[iii] += kp_out[28 + iii]
                
                for iii in range(5):
                    kp_3d_pts[start_idx + iii] = kp_3d_4d_[28 + 9 + iii]
                    kp_2d_pts[start_idx + iii] = kp_out[28 + 9 + iii]  
                start_idx += 5      
                
            for iii in range(9):
                kp_2d_pts[iii] /= valid_cnt                                      
                
            
            if valid_cnt > 0:
            
                valid_case += 1
            
                retval, rvec, tvec = cv2.solvePnP(
                                        objectPoints = np.ascontiguousarray(kp_3d_pts[:, 0:3].reshape((-1,1,3))), 
                                        imagePoints = np.ascontiguousarray(kp_2d_pts.reshape((-1,1,2))), 
                                        cameraMatrix = K,
                                        distCoeffs = np.zeros((8, 1), dtype=np.float64),
                                        flags=cv2.SOLVEPNP_ITERATIVE
                                        )                                                                                                                                                                
                R_est, _ = cv2.Rodrigues(rvec)                  
                
                pose_gt = {'R':rot_gt.data.cpu().numpy()[0], 't':tra_gt.data.cpu().numpy().reshape((3, -1))}
                pose = {'R':R_est, 't':tvec.reshape((3, -1))}

                vsd = PE.vsd(pose['R'], pose['t'], pose_gt['R'], pose_gt['t'], obj_depth[0], K, 15, [0.2], True, model_diameter, bop_renderer, int(args.obj), 'step')[0]  
                add = PE.add(pose['R'], pose['t'], pose_gt['R'], pose_gt['t'], obj_model.vertices) 
                adi = PE.adi(pose['R'], pose['t'], pose_gt['R'], pose_gt['t'], obj_model.vertices) 
                proj = PE.proj(pose['R'], pose['t'], pose_gt['R'], pose_gt['t'], K, obj_model.vertices) 
                re = PE.re(pose['R'], pose_gt['R'])
                te = PE.te(pose['t'], pose_gt['t'])        

                VSD_record.update(vsd, 1)
                ADD_record.update(add, 1)
                ADI_record.update(adi, 1)
                PROJ_record.update(proj, 1)
                RE_record.update(re, 1)
                TE_record.update(te, 1)  
                
                if vsd < 0.3:
                    true_vsd += 1                
                if add <= model_diameter*0.1: 
                    true_add += 1
                if adi <= model_diameter*0.1: 
                    true_adi += 1            
                if proj <= 5: 
                    true_proj += 1
                if re <= 5: 
                    true_re += 1
                if te <= 50: 
                    true_te += 1
        
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
                
        ################ pridiction ################
        primitive_x_np = (primitive_out[0, :3].data.permute(1, 2, 0).cpu().numpy()*255).copy()
        keypoint_x_np = keypoint_out_[0, :28].data.cpu().numpy().reshape((-1, 2))
        for iii in range(keypoint_x_np.shape[0]):
            primitive_x_np = cv2.circle(primitive_x_np, (int(keypoint_x_np[iii, 0]), int(keypoint_x_np[iii, 1])), 1, (255, 255, 255), -1)
        primitive_x_tensor = torch.from_numpy(primitive_x_np).permute(2, 0, 1).view(1, 3, 64, 64) / 255.
        
        primitive_y_np = (primitive_out[0, 3:6].data.permute(1, 2, 0).cpu().numpy()*255).copy()
        keypoint_y_np = keypoint_out_[0, 28:56].data.cpu().numpy().reshape((-1, 2))
        for iii in range(keypoint_y_np.shape[0]):
            primitive_y_np = cv2.circle(primitive_y_np, (int(keypoint_y_np[iii, 0]), int(keypoint_y_np[iii, 1])), 1, (255, 255, 255), -1)
        primitive_y_tensor = torch.from_numpy(primitive_y_np).permute(2, 0, 1).view(1, 3, 64, 64) / 255.
        
        primitive_z_np = (primitive_out[0, 6:9].data.permute(1, 2, 0).cpu().numpy()*255).copy()
        keypoint_z_np = keypoint_out_[0, 56:84].data.cpu().numpy().reshape((-1, 2))
        for iii in range(keypoint_z_np.shape[0]):
            primitive_z_np = cv2.circle(primitive_z_np, (int(keypoint_z_np[iii, 0]), int(keypoint_z_np[iii, 1])), 1, (255, 255, 255), -1)
        primitive_z_tensor = torch.from_numpy(primitive_z_np).permute(2, 0, 1).view(1, 3, 64, 64) / 255.               
                                 
        grid = torchvision.utils.make_grid(obj_inp, nrow=1)
        writer.add_image('val_images', grid, global_step=epoch*len(test_dataset) + i)     
        grid = torchvision.utils.make_grid(primitive_x_tensor, nrow=1)
        writer.add_image('val_primitive_x_out', grid, global_step=epoch*len(test_dataset) + i) 
        grid = torchvision.utils.make_grid(primitive_y_tensor, nrow=1)
        writer.add_image('val_primitive_y_out', grid, global_step=epoch*len(test_dataset) + i) 
        grid = torchvision.utils.make_grid(primitive_z_tensor, nrow=1)
        writer.add_image('val_primitive_z_out', grid, global_step=epoch*len(test_dataset) + i)
        ##############################################
         
    is_best_add = False
    is_best_adi = False
    is_best_vsd = False
    
    if len(test_dataset) - len(test_dataset)*0.05 <= valid_case:    
        if ADD_record.avg < best_add:
            best_add = ADD_record.avg
            is_best_add = True    
            
        if ADI_record.avg < best_adi:
            best_adi = ADI_record.avg
            is_best_adi = True         
            
        if VSD_record.avg < best_vsd:
            best_vsd = VSD_record.avg
            is_best_vsd = True                  
    
    progress_loss.print(len(test_dataset)) 
    
    print("----------------------------------------------------")
    print("valid case :", valid_case, "/", len(test_dataset))
    print(str(VSD_record), "\t", true_vsd/len(test_dataset)*100, "%")    
    print(str(ADD_record), "\t", true_add/len(test_dataset)*100, "%")
    print(str(ADI_record), "\t", true_adi/len(test_dataset)*100, "%")
    print(str(PROJ_record), "\t", true_proj/len(test_dataset)*100, "%")
    print(str(RE_record), "\t", true_re/len(test_dataset)*100, "%")
    print(str(TE_record), "\t", true_te/len(test_dataset)*100, "%")
        
    return is_best_vsd, is_best_add, is_best_adi


if __name__ == '__main__':
    main()

