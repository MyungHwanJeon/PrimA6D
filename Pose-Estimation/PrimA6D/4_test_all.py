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

from model import utils, Discriminator, TraModel, KeypointModel, ReconstModel, SegmentationNet
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
parser.add_argument('-w', '--downloaded_weight', dest='dw', action='store_true',
                     help='use downloaded weight')                                                             
                                                           
args = parser.parse_args()

os.makedirs('./checkpoints', exist_ok=True)   
os.makedirs('./checkpoints/' + args.dataset, exist_ok=True)

model_G_weight_path = "./checkpoints/" + args.dataset + "/obj_" + args.obj + "_G.pth"
model_K_weight_path = "./checkpoints/" + args.dataset + "/obj_" + args.obj + "_K.pth"
model_T_weight_path = "./checkpoints/" + args.dataset + "/obj_" + args.obj + "_T.pth"
if args.dw:
    model_G_weight_path = "./trained_weight/" + args.dataset + "/obj_" + args.obj + "_G.pth"
    model_K_weight_path = "./trained_weight/" + args.dataset + "/obj_" + args.obj + "_K.pth"
    model_T_weight_path = "./trained_weight/" + args.dataset + "/obj_" + args.obj + "_T.pth"
model_S_weight_path = "../Segmentation/trained_weight/" + args.dataset + "/obj_" + args.obj + "_S.pth"

obj_model_path = '../dataset/3d_model/' + str(args.dataset) + '/model_eval/obj_' +  "%06d" % int(args.obj) + '.ply'
obj_model = trimesh.load(obj_model_path)
obj_model_scale = 1.0
obj_model.vertices *= obj_model_scale
model_diameter = calc_pts_diameter2(obj_model.vertices)

primitive_model_path = '../dataset/3d_model/PRIMITIVE/3axis.ply'
primitive_model = trimesh.load(primitive_model_path) 
primitive_diameter = calc_pts_diameter2(primitive_model.vertices)
primitive_scale = model_diameter*0.6 / primitive_diameter
primitive_model.vertices *= primitive_scale

kp_3d_4d = np.array([
                    [0, 0, 0, 1],
                    [15, 15, 15, 1], [15, 15, -15, 1], [15, -15, 15, 1], [15, -15, -15, 1], [-15, 15, 15, 1], [-15, 15, -15, 1], [-15, -15, 15, 1], [-15, -15, -15, 1],
                    [105, 15, 15, 1], [105, 15, -15, 1], [105, -15, 15, 1], [105, -15, -15, 1],
                    [15, 105, 15, 1], [15, 105, -15, 1], [-15, 105, 15, 1], [-15, 105, -15, 1],
                    [15, 15, 105, 1], [15, -15, 105, 1], [-15, 15, 105, 1], [-15, -15, 105, 1],
                        ], dtype=np.float32)
kp_3d_4d[:, 0:3] = kp_3d_4d[:, 0:3] * primitive_scale

print("dataset : ", args.dataset)
print("dataset path : ", args.dataset_path)
print("obj : ", args.obj)

def main():    
            
    ## prepare test dataset ##
    test_file_path = os.path.join(args.dataset_path, "test/obj_%d"%int(args.obj))
    test_file_list = list()
    if os.path.isdir(test_file_path):
        test_file_list = [os.path.join(test_file_path, x) for x in os.listdir(test_file_path)] 
    test_file_list.sort()
    
    model_S = SegmentationNet.SegmentationNet(device=device).to(device)          
    model_S.cuda()        
    load_weight_all(model_S, pre_trained_path=model_S_weight_path)                             
    
    test_dataset = DatasetLoader(file_dir=test_file_list, train=False, primitive_scale=primitive_scale, img_origin=True, segmentation_model=model_S)
    test_dataset = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False)     
        
    model_G = ReconstModel.ReconstModel().to(device)
    model_G = nn.DataParallel(model_G)
    model_G.cuda()                           
    load_weight_all(model_G, pre_trained_path=model_G_weight_path)
    
    model_T = TraModel.TranslationModel().to(device)
    model_T = nn.DataParallel(model_T)
    model_T.cuda()         
    load_weight_all(model_T, pre_trained_path=model_T_weight_path)  
    
    model_K = KeypointModel.KeypointModel().to(device)
    model_K = nn.DataParallel(model_K)
    model_K.cuda()         
    load_weight_all(model_K, pre_trained_path=model_K_weight_path)

    test(test_dataset, model_G, model_T, model_K)

def test(test_dataset, model_G, model_T, model_K):

    batch_time = AverageMeter('Time', ':6.3f')
    add_record = AverageMeter('add', ':.4f')
    adi_record = AverageMeter('adi', ':.4f')     
    proj_record = AverageMeter('proj', ':.4f') 
    te_record = AverageMeter('te', ':.4f') 
    re_record = AverageMeter('re', ':.4f')   
               
    #switch to eval mode
    model_G.eval() 
    model_T.eval()   
    model_K.eval()     
    
    add_errs = []
    adi_errs = []
    true_proj = 0
    true_add = 0
    true_adi = 0    
    true_te = 0
    true_re = 0
    for i, data in enumerate(test_dataset):       
                
        img_origin, obj_inp, obj_gt, primitive_gt, obj_bb, rot_gt, tra_gt, kp_2d_gt, K, _ = data   
        obj_inp = obj_inp.to(device)
        K = K.data.cpu().numpy().reshape(3, 3)  
        
        end = time.time()  
        
        ## inference ##
        reconst_out, mu, sigma, primitive_out = model_G(obj_inp)
        keypoint_out = model_K(primitive_out)
        Tz, trans_input = model_T(reconst_out, obj_bb)
                
        keypoint_out_ = keypoint_out.clone()              
        kp_3d_4d_ = kp_3d_4d.reshape((21, 4))        
        
        ## rotation estimation ##
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
        
        ## translation estimation ##
        Tz = Tz[0].data.cpu().numpy()
        Tx = (kp_out[0, 0] - K[0, 2]) * Tz / K[0, 0]
        Ty = (kp_out[0, 1] - K[1, 2]) * Tz / K[1, 1]
        tvec = np.array([Tx, Ty, Tz], dtype=np.float32)
        
        ## evaluation ##                
        batch_time.update(time.time() - end)
        end = time.time() 
                
        pose_gt = {'R':rot_gt.data.cpu().numpy()[0], 't':tra_gt.data.cpu().numpy().reshape((3, -1))}
        pose = {'R':R_est, 't':tvec.reshape((3, -1))}
                        
        add_error = PE.add(pose['R'], pose['t'], pose_gt['R'], pose_gt['t'], obj_model.vertices) 
        adi_error = PE.adi(pose['R'], pose['t'], pose_gt['R'], pose_gt['t'], obj_model.vertices) 
        proj_error = PE.proj(pose['R'], pose['t'], pose_gt['R'], pose_gt['t'], K, obj_model.vertices) 
        rot_error = PE.re(pose['R'], pose_gt['R']) 
        tra_error = PE.te(pose['t'], pose_gt['t'])
        
        add_record.update(add_error, 1)
        adi_record.update(adi_error, 1)        
        proj_record.update(proj_error, 1)
        te_record.update(tra_error, 1)
        re_record.update(rot_error, 1)
        
        add_errs.append(add_error*0.001)        
        adi_errs.append(adi_error*0.001)
        
        if add_error <= model_diameter*0.1: 
            true_add += 1
        if adi_error <= model_diameter*0.1: 
            true_adi += 1            
        if proj_error <= 5: 
            true_proj += 1
        if rot_error <= 5: 
            true_re += 1
        if tra_error <= 50: 
            true_te += 1          
          
        ## visualization ##        
        model_info = json.load(open("../dataset/3d_model/" + args.dataset + "/models_info.json", 'r+'))
        minx = model_info[args.obj]["min_x"]
        miny = model_info[args.obj]["min_y"]
        minz = model_info[args.obj]["min_z"]
        sizex = model_info[args.obj]["size_x"]
        sizey = model_info[args.obj]["size_y"]
        sizez = model_info[args.obj]["size_z"]
        
        corner_pt = []
        corner_pt.append([minx, miny, minz])
        corner_pt.append([minx, miny, minz + sizez])
        corner_pt.append([minx, miny + sizey, minz])
        corner_pt.append([minx,miny + sizey, minz + sizez])
        corner_pt.append([minx + sizex, miny, minz])
        corner_pt.append([minx + sizex, miny, minz + sizez])
        corner_pt.append([minx + sizex, miny + sizey, minz])
        corner_pt.append([minx + sizex, miny + sizey, minz + sizez])
        corner_pt = np.array(corner_pt)
    
        img_origin = img_origin[0].detach().cpu().numpy()  
        
        proj_est = PE.misc.project_pts(corner_pt, K, pose_gt['R'], pose_gt['t'])                 
        pts = np.array([proj_est[0], proj_est[1], proj_est[3], proj_est[2], proj_est[0], proj_est[4], proj_est[6], proj_est[2]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        img_origin = cv2.polylines(img_origin, [pts], False, (0, 0, 255), 2)            
        pts = np.array([proj_est[5], proj_est[4], proj_est[6], proj_est[7], proj_est[5], proj_est[1], proj_est[3], proj_est[7]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        img_origin = cv2.polylines(img_origin, [pts], False, (0, 0, 255), 2)
                    
        proj_est = PE.misc.project_pts(corner_pt, K, pose['R'], pose['t'])            
        pts = np.array([proj_est[0], proj_est[1], proj_est[3], proj_est[2], proj_est[0], proj_est[4], proj_est[6], proj_est[2]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        img_origin = cv2.polylines(img_origin, [pts], True, (255, 0, 0), 2)            
        pts = np.array([proj_est[5], proj_est[4], proj_est[6], proj_est[7], proj_est[5], proj_est[1], proj_est[3], proj_est[7]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        img_origin = cv2.polylines(img_origin, [pts], True, (255, 0, 0), 2)
                
        primitive_np = (primitive_out[0].data.permute(1, 2, 0).cpu().numpy()*255).copy()
        keypoint_np = keypoint_out_[0].data.cpu().numpy().reshape((-1, 2))
        for iii in range(keypoint_np.shape[0]):
            primitive_np = cv2.circle(primitive_np, (int(keypoint_np[iii, 0]), int(keypoint_np[iii, 1])), 1, (255, 255, 255), -1)
        
        cv2.imshow("img_origin", img_origin)
        cv2.imshow("im", obj_inp[0].permute(1, 2, 0).data.cpu().numpy())
        cv2.imshow("primitivie_out", primitive_out[0].permute(1, 2, 0).data.cpu().numpy())
        cv2.imshow("primitivie_keypoint", primitive_np)
        cv2.imshow("reconst_out", reconst_out[0].permute(1, 2, 0).data.cpu().numpy())            
        cv2.waitKey(10)
 

    print("===================================================")    
    print("obj ", args.obj, " test result")    
    print("total : ", len(test_dataset))     
    print("time : ", batch_time.avg)    
    print("ADD_AUC:", compute_auc_posecnn(np.array(add_errs)))
    print("ADI_AUC:", compute_auc_posecnn(np.array(adi_errs)))    
    print("ADD < 0.1d: ", (true_add/len(test_dataset))*100., ", avg : ", add_record.avg)
    print("ADI < 0.1d: ", (true_adi/len(test_dataset))*100., ", avg : ", adi_record.avg)    
    print("Proj_2D < 5px: ", (true_proj/len(test_dataset))*100., ", avg : ", proj_record.avg)
    print("TE < 50mm: ", (true_te/len(test_dataset))*100., ", avg : ", te_record.avg)
    print("RE < 5deg: ", (true_re/len(test_dataset))*100., ", avg : ", re_record.avg)
    print("===================================================")  


if __name__ == '__main__':
    main()

