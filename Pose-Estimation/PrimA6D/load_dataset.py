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

import os
import random
import sys
import matplotlib.pyplot as plt
import copy
import gzip
import numpy as np
import cv2
import pickle
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import dataset.transform as transfrom
import imgaug
import imgaug.augmenters as iaa
import misc.randAugment as randAugment
from dataset.dataset_utils import *

class DatasetLoader(Dataset):

    def __init__(self, file_dir, train=True, primitive_scale=1, img_origin=False, segmentation_model=None, device=torch.device('cuda')):
        
        self.file_dir = file_dir
        self.num_files = len(file_dir)
        self.img_origin = img_origin
        self.primitive_scale = primitive_scale
        self.train = train
        self.device = device     
        self.segmentation_model=segmentation_model 
        
        if self.segmentation_model is not None:
            self.segmentation_model.cuda()
            self.segmentation_model.eval()                  
        
        self.kp_3d_4d = np.array([
                    [0, 0, 0, 1],
                    [15, 15, 15, 1], [15, 15, -15, 1], [15, -15, 15, 1], [15, -15, -15, 1], [-15, 15, 15, 1], [-15, 15, -15, 1], [-15, -15, 15, 1], [-15, -15, -15, 1],
                    [105, 15, 15, 1], [105, 15, -15, 1], [105, -15, 15, 1], [105, -15, -15, 1],
                    [15, 105, 15, 1], [15, 105, -15, 1], [-15, 105, 15, 1], [-15, 105, -15, 1],
                    [15, 15, 105, 1], [15, -15, 105, 1], [-15, 15, 105, 1], [-15, -15, 105, 1],
                        ], dtype=np.float32)
        self.kp_3d_4d[:, 0:3] = self.kp_3d_4d[:, 0:3] * self.primitive_scale                          
   
        self.ToTensor = transforms.Compose([
                                transforms.ToTensor()                                     
                                ])                                                      
                                
        #self.aug_arithmetic = iaa.Sequential([
        #                    iaa.Sometimes(0.5, iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False)),
        #                    iaa.Sometimes(0.5, iaa.Cutout(fill_mode="constant", cval=(0, 255), fill_per_channel=0.5)),
        #                    iaa.Sometimes(0.5, iaa.CoarseDropout((0.0, 0.2), size_percent=(0.02, 0.25)))
        #                    ], random_order=False)    
                            
        self.aug_arithmetic = iaa.Sequential([
                            iaa.Sometimes(0.5, iaa.Cutout(nb_iterations=(1, 5), size=0.3, squared=False)),         
                            iaa.Sometimes(0.5, iaa.CoarseDropout((0.1, 0.5), size_percent=(0.01, 0.05)))
                            ], random_order=False)                                
                                                                                                                                          
                            
        self.rand_augment = randAugment.RandAugment(n = 2, m = 9)        
                                   
    def __len__(self):
        return len(self.file_dir)
        
    def __getitem__(self, index):
                
        f = gzip.open(self.file_dir[index], "rb")
        data = pickle.load(f)                
                
        obj_inp = data["obj_inp"].astype(np.uint8)
        obj_mask = data["obj_mask"].astype(np.uint8)
        obj_gt = data["obj_gt"].astype(np.uint8)       
        primitive_gt = data["primitive_gt"].astype(np.uint8)
        #primitive_x_gt = data["primitive_x_gt"].astype(np.uint8)
        #primitive_y_gt = data["primitive_y_gt"].astype(np.uint8)
        #primitive_z_gt = data["primitive_z_gt"].astype(np.uint8)       
        rot_gt = data["rot_gt"].astype(np.float32)
        tra_gt = data["tra_gt"].astype(np.float32)       
        K = data["k"].astype(np.float32)               
                
        if 'obj_depth' in data.keys():   
            obj_depth = data["obj_depth"].astype(np.float32)
        
        if self.segmentation_model is not None:

            segmentation_inp = self.ToTensor(obj_inp)
            segmentation_inp = segmentation_inp.view(1, segmentation_inp.shape[0], segmentation_inp.shape[1], segmentation_inp.shape[2])
            segmentation_pred = self.segmentation_model(segmentation_inp.to(self.device))
            segmentation_pred = torch.argmax(segmentation_pred, dim=1)          

            obj_mask = segmentation_pred.view(480, 640, 1).data.cpu().numpy().astype(np.uint8)        
            obj_idx = torch.nonzero(segmentation_pred.view(480, 640))   
            obj_bb_xywh = calc_2d_bbox(obj_idx[:, 1], obj_idx[:, 0], [640, 480, 3])        
            
        else:
            obj_ys, obj_xs = np.nonzero(obj_mask > 0)
            obj_bb_xywh = calc_2d_bbox(obj_xs, obj_ys, [obj_inp.shape[1], obj_inp.shape[0], obj_inp.shape[2]])
            
            
        x, y, w, h = obj_bb_xywh  
    
        aug = random.randint(0, 10)
        if self.train and aug > 0:
        
            crop_ratio = np.random.rand(1)
            if crop_ratio > 0.7:
                crop_ratio = 0.7
            crop_axis = np.random.randint(0, 2)
            crop_direction = np.random.randint(0, 2)
            
            if crop_axis == 0:
                if crop_direction == 0:
                    w = w - w * crop_ratio          
                if crop_direction == 1:
                    x = x + w * crop_ratio
                    w = w - w * crop_ratio   
            if crop_axis == 1:
                if crop_direction == 0:
                    h = h - h * crop_ratio          
                if crop_direction == 1:
                    y = y + h * crop_ratio
                    h = h - h * crop_ratio  

            obj_bb_xywh = np.array([x, y, w, h], dtype=np.uint32)

        aug = random.randint(0, 10)
        if self.train and aug > 0:
            max_rel_offset = 0.2
            rand_trans_x = np.random.uniform(-max_rel_offset, max_rel_offset) * obj_bb_xywh[2]
            rand_trans_y = np.random.uniform(-max_rel_offset, max_rel_offset) * obj_bb_xywh[3]
            obj_bb_xywh = obj_bb_xywh + np.array([rand_trans_x,rand_trans_y, -rand_trans_x, -rand_trans_y], dtype=np.uint32)
    

        cropped_obj_inp, obj_bb_square_xywh = crop_image_using_bb(img=obj_inp, bb_xywh=obj_bb_xywh)        
        padding_obj_bb_xywh = make_padding_bb(img=cropped_obj_inp, bb_xywh=obj_bb_square_xywh, render_dimension=[obj_inp.shape[1], obj_inp.shape[0], obj_inp.shape[2]], pad_factor=1.3)
        
        bb_size = int(np.maximum(padding_obj_bb_xywh[2], padding_obj_bb_xywh[3]))
        blank_image = np.zeros((int(bb_size), int(bb_size), 3), dtype=np.uint8) 

        left = np.maximum(obj_bb_square_xywh[0] - padding_obj_bb_xywh[0], 0)
        right = np.minimum(left + obj_bb_square_xywh[2], bb_size)
        top = np.maximum(obj_bb_square_xywh[1] - padding_obj_bb_xywh[1], 0)
        bottom = np.minimum(top + obj_bb_square_xywh[3], bb_size)
        
        cropped_obj_inp = cv2.resize(cropped_obj_inp, (int(right)-int(left), int(bottom)-int(top)), interpolation=cv2.INTER_NEAREST)         
        blank_image[int(top):int(bottom), int(left):int(right)] = cropped_obj_inp

        left = padding_obj_bb_xywh[0]
        right = padding_obj_bb_xywh[0] + padding_obj_bb_xywh[2]
        top = padding_obj_bb_xywh[1]
        bottom = padding_obj_bb_xywh[1] + padding_obj_bb_xywh[3]
        blank_image = blank_image[0:int(bottom)-int(top), 0:int(right)-int(left)]
        obj_mask = obj_mask[int(top):int(bottom), int(left):int(right)]
        primitive_gt = primitive_gt[int(top):int(bottom), int(left):int(right)]
        obj_gt = obj_gt[int(top):int(bottom), int(left):int(right)]
               
        result_image_size = (64, 64)
        if padding_obj_bb_xywh[2] > result_image_size[0] or padding_obj_bb_xywh[3] > result_image_size[0]:
            obj_inp = cv2.resize(blank_image, result_image_size, interpolation=cv2.INTER_AREA) 
            obj_mask = cv2.resize(obj_mask, result_image_size, interpolation=cv2.INTER_AREA)         
            primitive_gt = cv2.resize(primitive_gt, result_image_size, interpolation=cv2.INTER_AREA)           
            obj_gt = cv2.resize(obj_gt, result_image_size, interpolation=cv2.INTER_AREA)
            
        else:
            obj_inp = cv2.resize(blank_image, result_image_size, interpolation=cv2.INTER_LINEAR_EXACT)
            obj_mask = cv2.resize(obj_mask, result_image_size, interpolation=cv2.INTER_LINEAR_EXACT)
            primitive_gt = cv2.resize(primitive_gt, result_image_size, interpolation=cv2.INTER_LINEAR_EXACT)                       
            obj_gt = cv2.resize(obj_gt, result_image_size, interpolation=cv2.INTER_LINEAR_EXACT)                
                        
        aug = random.randint(0, 10)
        if self.train and aug > 0:
        
            padding_mask = obj_inp[:, :, 0] == 0
            obj_inp = obj_inp.reshape((1,)+obj_inp.shape)
            
            obj_inp_aug = self.aug_arithmetic(images=obj_inp)
            diff = np.abs(obj_inp_aug - obj_inp)
            diff = diff > 0
            diff = diff.reshape((64, 64, 3))
            obj_mask[diff[:, :, 0]] = 0
            obj_mask[diff[:, :, 1]] = 0
            obj_mask[diff[:, :, 2]] = 0
               
            obj_inp = self.rand_augment.augment_images(obj_inp_aug).reshape((64, 64, 3))
            obj_inp[padding_mask] = np.zeros((64, 64, 3))[padding_mask]    
             
                    
        R_t = np.zeros((3, 4), dtype=np.float32)                                         
        R_t[0:3, 0:3,] = rot_gt
        R_t[0:3, 3] = tra_gt
        P = np.matmul(K, R_t)
        kp_2d = np.matmul(P, np.transpose(self.kp_3d_4d))
        
        kp_2d[0, :] = kp_2d[0, :] / kp_2d[2, :]
        kp_2d[1, :] = kp_2d[1, :] / kp_2d[2, :]
        kp_2d[2, :] = kp_2d[2, :] / kp_2d[2, :]
        kp_2d = np.transpose(kp_2d)                        

        kp_2d[:, 0] = (kp_2d[:, 0]-padding_obj_bb_xywh[0]) * (64. / padding_obj_bb_xywh[2])
        kp_2d[:, 1] = (kp_2d[:, 1]-padding_obj_bb_xywh[1]) * (64. / padding_obj_bb_xywh[3])

        kp_2d_gt = kp_2d[:, 0:2]#.reshape(-1)                                                           
                
        obj_inp = self.ToTensor(obj_inp)        
        obj_gt = self.ToTensor(obj_gt)
        primitive_gt = self.ToTensor(primitive_gt)
        obj_bb = padding_obj_bb_xywh      
        
        if self.img_origin:
            if 'obj_depth' in data.keys(): 
                return data["obj_inp"].astype(np.uint8), obj_inp, obj_gt, primitive_gt, obj_bb, rot_gt, tra_gt, kp_2d_gt, K, obj_depth
            else:
                return data["obj_inp"].astype(np.uint8), obj_inp, obj_gt, primitive_gt, obj_bb, rot_gt, tra_gt, kp_2d_gt, K, np.ones(1)
        else:
            if 'obj_depth' in data.keys(): 
                return obj_inp, obj_gt, primitive_gt, obj_bb, rot_gt, tra_gt, kp_2d_gt, K, obj_depth
            else:
                return obj_inp, obj_gt, primitive_gt, obj_bb, rot_gt, tra_gt, kp_2d_gt, K, np.ones(1)
