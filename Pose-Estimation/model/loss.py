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
#import torchvision.utils as utils

from model import utils

import os
import random
import shutil
import time
import warnings
import sys
import matplotlib.pyplot as plt
import copy
import numpy as np
import cv2

def MSE_loss(pred, gt, reduction='mean'):
 
    loss = nn.MSELoss(reduction=reduction)(pred, gt)

    return loss
     
def MAE_loss(pred, gt, reduction='mean'):
 
    loss = nn.L1Loss(reduction=reduction)(pred, gt)

    return loss       
            
def BCE_loss(pred, is_real=False, device=torch.device('cuda')):
    
    target_tensor =  torch.tensor(0.0)
    if is_real:
        target_tensor =  torch.tensor(1.0)

    target_tensor = target_tensor.expand_as(pred)    
    loss = nn.BCELoss()(pred, target_tensor.to(device))            
    
    return loss
    
def CE_loss(pred, gt):
    
    loss = nn.CrossEntropyLoss()(pred, gt)           
    
    return loss       
    
def kl_divgence_loss(mu, sigma, device=torch.device('cuda')):

    normal_distribution = torch.distributions.normal.Normal(torch.zeros(mu.shape), torch.ones(sigma.shape))
    target_distribution = normal_distribution.sample().to(device)
    distribution = torch.distributions.normal.Normal(mu, sigma)
    current_distribution = distribution.sample().to(device)   
    
    kl_loss = F.kl_div(current_distribution, target_distribution, reduction='batchmean')

    return kl_loss  
    
def Top_K_MSE_loss(pred, gt, k=4):
    
    pred_flat = torch.flatten(pred, start_dim=1)        
    gt_flat = torch.flatten(gt, start_dim=1)        
    loss = nn.MSELoss(reduction='none')(pred_flat, gt_flat)
    values, indices = torch.topk(loss, int(loss.shape[1]/k), dim=1)       
    loss = torch.mean(values)

    return loss        

def Top_K_MAE_loss(pred, gt, k=4):
    
    pred_flat = torch.flatten(pred, start_dim=1)        
    gt_flat = torch.flatten(gt, start_dim=1)        
    loss = nn.L1Loss(reduction='none')(pred_flat, gt_flat)
    values, indices = torch.topk(loss, int(loss.shape[1]/k), dim=1)       
    loss = torch.mean(values)

    return loss       

def get_covariance_fully_independent(cov_vec):

    batch_size = cov_vec.shape[0]

    D_size = cov_vec.shape[1]
    D = torch.zeros((batch_size, D_size, D_size)).cuda()
    diag_indices = np.diag_indices(D.shape[1])
    #D[:, diag_indices[0], diag_indices[1]] = torch.exp(cov_vec[:,]) #no non-linear activation as the final activation function
    D[:, diag_indices[0], diag_indices[1]] = cov_vec[:,] + 0.000001 #using ReLU or Softplus or Sigmoid as the final activation fuction
    
    covariance = D
    
    return covariance    
        
def keypoint_uncertainty_loss(pred, gt, uncertainty):
 
    batch_size = pred.shape[0]
    uncertainty = uncertainty.view(-1, 1)

    error = gt - pred    
    error = error.reshape((batch_size, error.shape[1])).cuda()    
    error_T = error.transpose(0, 1).cuda()        
        
    loss = torch.mean(error * (1/torch.exp(uncertainty)) * error + torch.exp(uncertainty)*28)
                   
    return loss.sum()    
    
def quaternion_loss(pred, gt):
    
    ## 2*acos(sum(|pred*gt|))       

    pred = F.normalize(pred)
    gt = F.normalize(gt)
    
    loss = torch.sum(pred * gt, dim=1, keepdim=True)
    loss = torch.abs(loss)
    loss = torch.acos(loss)   
    loss = 2 * loss
    loss = torch.mean(loss)    
    
    return loss  
    
def geodedic_loss(pred, gt, reduction='mean'):

    batch=pred.shape[0]

    m = torch.bmm(pred, gt.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
        
    theta = torch.acos(cos)
    
    theta = (180.0 * theta) / np.pi
    
    if reduction == 'mean':
        theta = torch.mean(theta) 
            
    return theta
    
    
def rotation_primitive_loss(pred, gt):

    ## measure the misalignment per channel ##
    const_A = 5
    pred_flat = torch.flatten(pred[:, 0, :, :], start_dim=1)  
    gt_flat = torch.flatten(gt[:, 0, :, :], start_dim=1)  
            
    diff_A = torch.abs(pred_flat - gt_flat)
    weight_A = torch.exp(const_A * diff_A)
    loss_A = diff_A * diff_A * weight_A
    values_A, indices_A = torch.topk(loss_A, int(loss_A.shape[1]/4), dim=1)
    loss_A = torch.mean(values_A, dim=1)    
    
    const_B = 5
    pred_flat = torch.flatten(pred[:, 1, :, :], start_dim=1)  
    gt_flat = torch.flatten(gt[:, 1, :, :], start_dim=1) 
    
    diff_B = torch.abs(pred_flat - gt_flat)
    weight_B = torch.exp(const_B * diff_B)
    loss_B = diff_B * diff_B * weight_B
    values_B, indices_B = torch.topk(loss_B, int(loss_B.shape[1]/4), dim=1)
    loss_B = torch.mean(values_B, dim=1)   
        
    const_C = 5
    pred_flat = torch.flatten(pred[:, 2, :, :], start_dim=1)  
    gt_flat = torch.flatten(gt[:, 2, :, :], start_dim=1) 
    
    diff_C = torch.abs(pred_flat - gt_flat)
    weight_C = torch.exp(const_C * diff_C)
    loss_C = diff_C * diff_C * weight_C
    values_C, indices_C = torch.topk(loss_C, int(loss_C.shape[1]/4), dim=1)
    loss_C = torch.mean(values_C, dim=1)   
                        
    ## measure the misalignment btw channels ##     
    values_A, indices_A = torch.topk(diff_A, k=128, dim=1)
    diff_A_sum = torch.sum(values_A, dim=1)
    values_B, indices_B = torch.topk(diff_B, k=128, dim=1)
    diff_B_sum = torch.sum(values_B, dim=1)
    values_C, indices_C = torch.topk(diff_C, k=128, dim=1)
    diff_C_sum = torch.sum(values_C, dim=1)
          
    diff_all_sum = diff_A_sum + diff_B_sum + diff_C_sum        
    ch_weight_A = torch.exp((diff_A_sum / diff_all_sum)*5)
    ch_weight_B = torch.exp((diff_B_sum / diff_all_sum)*5)
    ch_weight_C = torch.exp((diff_C_sum / diff_all_sum)*5)        
    
    loss = torch.mean(ch_weight_A*loss_A + ch_weight_B*loss_B + ch_weight_C*loss_C)
    
    return loss                 

def rotation_loss(pred_R, gt_R, model_pts):
             
    model_pts = torch.from_numpy(model_pts.astype(np.float32)).cuda()
    n_model_pts = model_pts.shape[0]
    bs = pred_R.shape[0]
    model_pts = model_pts.reshape(n_model_pts, 3)
    model_pts = model_pts.view(1, n_model_pts, 3).repeat(bs, 1, 1).permute(0, 2, 1)    
    
    pred_R = pred_R.reshape(bs, 3, 3)
    pred_model_pts = torch.bmm(pred_R, model_pts).cuda()

    gt_R = gt_R.reshape(bs, 3, 3)    
    gt_model_pts = torch.bmm(gt_R, model_pts).cuda()

    loss = nn.MSELoss(reduction='mean')(gt_model_pts, pred_model_pts) 
        
    return loss        

def translation_loss(pred_t, gt_t):
         
    loss = torch.mean(torch.sqrt((pred_t - gt_t)**2))
         
    return loss  

def transformation_loss(pred_R, pred_t, gt_R, gt_t, model_pts):
         
    model_pts = torch.from_numpy(model_pts.astype(np.float32)).cuda()
    n_model_pts = model_pts.shape[0]
    bs = pred_R.shape[0]
    model_pts = model_pts.reshape(n_model_pts, 3)
    model_pts = torch.cat((model_pts, torch.ones(n_model_pts, 1).cuda()), dim=-1)
    model_pts = model_pts.view(1, n_model_pts, 4).repeat(bs, 1, 1).permute(0, 2, 1)    
    
    pred_R = pred_R.reshape(bs, 3, 3)  
    pred_T = torch.cat((pred_R[:,0:3,0:3], pred_t.view(bs, 3, 1)), dim=-1)
    pred_model_pts = torch.bmm(pred_T, model_pts).cuda()

    gt_R = gt_R.reshape(bs, 3, 3)     
    gt_T = torch.cat((gt_R[:,0:3,0:3], gt_t.view(bs, 3, 1)), dim=-1)
    gt_model_pts = torch.bmm(gt_T, model_pts).cuda()

    loss = nn.MSELoss(reduction='mean')(gt_model_pts, pred_model_pts) 
        
    return loss    

             
class LogLoss(nn.Module):
    def __init__(self, use_gpu = True):
        super(LogLoss, self).__init__()
        self.log_block = LogEachBlock()
        self.loss = nn.L1Loss()
        if use_gpu:
            self.log_block.cuda()

    def __call__(self, input_A, input_B):
        log_A = self.log_block(input_A)
        log_B = self.log_block(input_B)
        return self.loss(log_A, log_B)


class LogBlock(nn.Module):
    """LoG Filter Block for LoG loss"""
    def __init__(self):
        super(LogBlock, self).__init__()

        ## RGB to Gray Block
        np_filter1 = np.array([[0.2989, 0.5870, 0.1140]]);
        conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False)
        conv1.weight = nn.Parameter(torch.from_numpy(np_filter1).float().unsqueeze(2).unsqueeze(2))

        ## LoG Filter Block
        np_filter2=np.array([[0, -1, 0],[-1,4,-1],[0,-1,0]])
        conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv2.weight=nn.Parameter(torch.from_numpy(np_filter2).float().unsqueeze(0).unsqueeze(0))

        self.main = nn.Sequential(conv1, conv2)

        ## Fix all weights
        for param in self.main.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.main(x)

class LogEachBlock(nn.Module):
    """LoG Filter Block for LoG Loss, applied for each color channel"""
    def __init__(self):
        super(LogEachBlock, self).__init__()

        ## LoG Filter Block
        np_filter2=np.array([[0, -1, 0],[-1,4,-1],[0,-1,0]])
        self.conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2.weight=nn.Parameter(torch.from_numpy(np_filter2).float().unsqueeze(0).unsqueeze(0))

        self.avg_pool = torch.nn.AvgPool3d((3,1,1), stride=1)
        self.smooth = torch.nn.AvgPool2d(3, stride=1, padding=1)

        for param in self.conv2.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Split each channel
        chan1 = x[:,0,:,:]
        chan2 = x[:,1,:,:]
        chan3 = x[:,2,:,:]

        # Match dimension batch x chan x height x width
        chan1 = chan1.unsqueeze(1)
        chan2 = chan2.unsqueeze(1)
        chan3 = chan3.unsqueeze(1)

        # chan1 = self.smooth(chan1)
        # chan2 = self.smooth(chan2)
        # chan3 = self.smooth(chan3)

        filt1 = self.conv2(chan1)
        filt2 = self.conv2(chan2)
        filt3 = self.conv2(chan3)

        # Concat channels and apply pooling
        concat = torch.cat((filt1, filt2, filt3), 1)
        output = self.avg_pool(concat)
        return output             
 
