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

from torchsummary import summary

import argparse
import os
import random
import shutil
import time
import warnings
import sys
import matplotlib
matplotlib.use('TKAgg') 
import matplotlib.pyplot as plt
import copy
import math
from PIL import Image
from scipy.spatial import distance


import numpy as np

import cv2

import pickle

#import soft_renderer as sr

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

def load_weight_all(model, pre_trained_path="", device=torch.device('cuda')):
        
    if os.path.exists(pre_trained_path):
        checkpoint = torch.load(pre_trained_path, map_location=device)
        state_dict = checkpoint["state_dict"]
        #print(state_dict.keys())
        model.load_state_dict(state_dict)        
        return True
    else:
        print("no exist : ", pre_trained_path)        
        return False
    
def load_weight_partially(model, pre_trained_path="", module=""):
        
    if os.path.exists(pre_trained_path):
        checkpoint = torch.load(pre_trained_path)
        state_dict = checkpoint["state_dict"]
        #print(state_dict.keys())
        pretrained_dict = {k: v for k, v in state_dict.items() if module in k}
        
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return True
    else:
        print("no exist : ", pre_trained_path)
        return False
            
def reset_trainable_all(model):

    for name, param in model.named_parameters():
        param.requires_grad = False  
        
            
def reset_trainable(model, module=""):

    for name, param in model.named_parameters():
        if module in name:
            param.requires_grad = False       
        
def set_trainable(model, module=""):
    
    for name, param in model.named_parameters():
        if module in name:
            param.requires_grad = True    
            
def set_trainable_all(model):
    
    for name, param in model.named_parameters():
        param.requires_grad = True    
    

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename + '_best')
        
def adjust_learning_rate(optimizer, epoch, learning_rate, cp):

    lr = learning_rate * (0.5 ** (epoch // cp))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
                
def calc_pts_diameter(pts):
    """
    Calculates diameter of a set of points (i.e. the maximum distance between
    any two points in the set).

    :param pts: nx3 ndarray with 3D points.
    :return: Diameter.
    """
    diameter = -1
    for pt_id in range(pts.shape[0]):
        #if pt_id % 1000 == 0: print(pt_id)
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter

def calc_pts_diameter2(pts):
    """
    Calculates diameter of a set of points (i.e. the maximum distance between
    any two points in the set). Faster but requires more memory than
    calc_pts_diameter.

    :param pts: nx3 ndarray with 3D points.
    :return: Diameter.
    """
    dists = distance.cdist(pts, pts, 'euclidean')
    diameter = np.max(dists)
    return diameter            
    
def compute_auc_posecnn(errors):
    # NOTE: Adapted from https://github.com/yuxng/YCB_Video_toolbox/blob/master/evaluate_poses_keyframe.m
    errors = errors.copy()
    d = np.sort(errors)
    #print(d)
    d[d > 0.1] = np.inf
    #print(d)
    accuracy = np.cumsum(np.ones(d.shape[0])) / d.shape[0]       
    ids = np.isfinite(d)
    #print(ids)
    d = d[ids]
    accuracy = accuracy[ids]
    if len(ids) == 0 or ids.sum() == 0:
        return np.nan
    rec = d
    prec = accuracy
    mrec = np.concatenate(([0], rec, [0.1]))
    mpre = np.concatenate(([0], prec, [prec[-1]]))
    #print(mrec)
    for i in np.arange(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.arange(1, len(mpre))
    ids = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = ((mrec[ids] - mrec[ids-1]) * mpre[ids]).sum() * 10
    return ap        
                
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def plot_unc(unc):

    axis = [""]
    fig = plt.figure(figsize=(1,4), facecolor='black')
    ax = plt.axes()
    ax.set_facecolor("black")
    plt.ylim(0, 1.0)
    plt.xlim(-1.0, 1.0)                    
    plt.bar(axis, unc, width=0.8)
    plt.xlabel("%.3f"%unc, fontdict={'color': 'white', 'weight': 'bold', 'size': 15})
    plt.hlines(0.4, -1, 1, color='red', linestyle='solid', linewidth=2)    

    fig.canvas.draw()
    unc_fig = np.array(fig.canvas.renderer._renderer)        
    unc_fig = cv2.cvtColor(unc_fig, cv2.COLOR_BGR2RGB)
    plt.close(fig)
    
    #unc_fig = np.ones((40, 200, 3))

    return unc_fig                
