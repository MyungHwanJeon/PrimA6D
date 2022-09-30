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


class DatasetLoader(Dataset):

    def __init__(self, file_dir, train=True):
        self.file_dir = file_dir
        self.num_files = len(file_dir)
        self.train = train        
            
        self.ToTensor = transforms.Compose([
                                transforms.ToTensor()                                     
                                ])                                                                                                  
                            
        self.rand_augment = randAugment.RandAugment(n = 2, m = 9)
        
        self.geometrical_aug = iaa.Sequential([
                            iaa.Sometimes(0.5, iaa.Dropout([0.05, 0.2])),  
                            iaa.Sometimes(0.5, iaa.Affine(rotate=(-45, 45))),  
                            
                            iaa.Sometimes(0.5, iaa.Rot90((1, 3))),  
                            iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.2))),  
                            iaa.Sometimes(0.5, iaa.ShearY((-20, 20))),  
                            iaa.Sometimes(0.5, iaa.ShearX((-20, 20))),  
                            
                            iaa.Sometimes(0.5, iaa.Rotate((-45, 45))),  
                            iaa.Sometimes(0.5, iaa.TranslateY(percent=(-0.2, 0.2))),  
                            iaa.Sometimes(0.5, iaa.TranslateX(percent=(-0.2, 0.2))),  
                            iaa.Sometimes(0.5, iaa.ScaleY((0.5, 1.5))),  
                            iaa.Sometimes(0.5, iaa.ScaleX((0.5, 1.5))),                              
                            
                            ], random_order=True)                

    def __len__(self):
        return len(self.file_dir)
        
    def __getitem__(self, index):
                
        f = gzip.open(self.file_dir[index], "rb")
        data = pickle.load(f)        
        
        obj_inp = data["obj_inp"].astype(np.uint8)
        obj_mask = data["obj_mask"].astype(np.uint8)

        obj_inp = obj_inp.reshape((480, 640, 3))  
        obj_mask = obj_mask.reshape((480, 640, 1))
        obj_mask = imgaug.augmentables.segmaps.SegmentationMapsOnImage(obj_mask, shape=obj_inp.shape)
                
        if self.train:
            obj_inp, obj_mask = self.geometrical_aug(image=obj_inp, segmentation_maps=obj_mask)  
            aug = random.randint(0, 2)      
            if aug > 0:
                obj_inp = self.rand_augment.augment_images(obj_inp.reshape((1, 480, 640, 3)))
            obj_inp = obj_inp.reshape((480, 640, 3))

        obj_inp = self.ToTensor(obj_inp.copy())
        obj_mask = obj_mask.get_arr().reshape((480, 640)) / 255.
        
        return obj_inp, obj_mask
