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

from model import utils, Discriminator, PrimA6DPPNet, SegmentationNet
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
parser.add_argument('-w', '--downloaded_weight', dest='dw', action='store_true',
                     help='use downloaded weight')                                                                                                     
                                                           
args = parser.parse_args()

os.makedirs('./checkpoints', exist_ok=True)   
os.makedirs('./checkpoints/' + args.dataset, exist_ok=True)



model_P_weight_path = "./checkpoints/" + args.dataset + "/obj_" + args.obj + "_P.pth"
if args.dw:
    model_P_weight_path = "./trained_weight/" + args.dataset + "/obj_" + args.obj + "_P.pth"
model_S_weight_path = "../Segmentation/trained_weight/" + args.dataset + "/obj_" + args.obj + "_S.pth"

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
kp_3d_4d[:, :, 0:3] = kp_3d_4d[:, :, 0:3] * primitive_scale

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
                
    ## prepare model ##   
    model_P = PrimA6DPPNet.PrimA6DNet(device=device).to(device)
    model_P.cuda()                          
    load_weight_all(model_P, pre_trained_path=model_P_weight_path)              
 
    test(test_dataset, model_P)       
            
def test(test_dataset, model_P):

    batch_time = AverageMeter('Time', ':6.3f')
    VSD_record = AverageMeter('VSD < 0.3', ':.4f')
    ADD_record = AverageMeter('ADD < 0.1d', ':.4f')
    ADI_record = AverageMeter('ADI < 0.1d', ':.4f')    
    PROJ_record = AverageMeter('PROJ < 5px', ':.4f')
    RE_record = AverageMeter('RE < 5deg', ':.4f')
    TE_record = AverageMeter('TE < 50mm', ':.4f')    
        
    #switch to eval mode
    model_P.eval() 

    add_errs = []
    adi_errs = []
    true_vsd = 0    
    true_add = 0
    true_adi = 0    
    true_proj = 0
    true_re = 0
    true_te = 0 
    
    valid_case = 0
    
    end = time.time()  
    for i, data in enumerate(test_dataset):
    
        img_origin, obj_inp, obj_gt, primitive_gt, primitive_x_gt, primitive_y_gt, primitive_z_gt, obj_bb, rot_gt, tra_gt, kp_2d_gt, K, obj_depth = data      
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
                
                add_errs.append(add*0.001)        
                adi_errs.append(adi*0.001)
                
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
        
        
                ################ visualization ################
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
                
                primitive_x_np = (primitive_out[0, :3].data.permute(1, 2, 0).cpu().numpy()).copy()
                keypoint_x_np = keypoint_out_[0, :28].data.cpu().numpy().reshape((-1, 2))
                for iii in range(keypoint_x_np.shape[0]):
                    primitive_x_np = cv2.circle(primitive_x_np, (int(keypoint_x_np[iii, 0]), int(keypoint_x_np[iii, 1])), 1, (1, 1, 1), -1)             
                
                primitive_y_np = (primitive_out[0, 3:6].data.permute(1, 2, 0).cpu().numpy()).copy()
                keypoint_y_np = keypoint_out_[0, 28:56].data.cpu().numpy().reshape((-1, 2))
                for iii in range(keypoint_y_np.shape[0]):
                    primitive_y_np = cv2.circle(primitive_y_np, (int(keypoint_y_np[iii, 0]), int(keypoint_y_np[iii, 1])), 1, (1, 1, 1), -1)  
                
                primitive_z_np = (primitive_out[0, 6:9].data.permute(1, 2, 0).cpu().numpy()).copy()
                keypoint_z_np = keypoint_out_[0, 56:84].data.cpu().numpy().reshape((-1, 2))
                for iii in range(keypoint_z_np.shape[0]):
                    primitive_z_np = cv2.circle(primitive_z_np, (int(keypoint_z_np[iii, 0]), int(keypoint_z_np[iii, 1])), 1, (1, 1, 1), -1)

                unc_x_fig = plot_unc((torch.exp(keypoint_uncertainty_out[0, 0])).item())              
                unc_y_fig = plot_unc((torch.exp(keypoint_uncertainty_out[0, 1])).item())
                unc_z_fig = plot_unc((torch.exp(keypoint_uncertainty_out[0, 2])).item())

                unc_x_fig = cv2.resize(unc_x_fig, (40, 160))
                unc_y_fig = cv2.resize(unc_y_fig, (40, 160))
                unc_z_fig = cv2.resize(unc_z_fig, (40, 160))

                primitive_x_np = cv2.resize(primitive_x_np, (160, 160))
                primitive_y_np = cv2.resize(primitive_y_np, (160, 160))
                primitive_z_np = cv2.resize(primitive_z_np, (160, 160))
                
                total_img = np.zeros((480, 840, 3), dtype=np.float32)
                total_img[:480, :640, :] = img_origin/255.
                total_img[0:160, 640:800, :] = primitive_x_np
                total_img[160:320, 640:800, :] = primitive_y_np
                total_img[320:480, 640:800, :] = primitive_z_np
                total_img[0:160, 800:840, :] = unc_x_fig/255.
                total_img[160:320, 800:840, :] = unc_y_fig/255.
                total_img[320:480, 800:840, :] = unc_z_fig/255.

                cv2.imshow("result", total_img)                   
                cv2.waitKey(1)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
                                  
    print("===================================================")    
    print("obj ", args.obj, " test result")    
    print("valid case :", valid_case, "/", len(test_dataset))
    print("ADD_AUC:", compute_auc_posecnn(np.array(add_errs)))    
    print("ADI_AUC:", compute_auc_posecnn(np.array(adi_errs)))
    print(str(VSD_record), "\t", true_vsd/len(test_dataset)*100, "%")
    print(str(ADD_record), "\t", true_add/len(test_dataset)*100, "%")    
    print(str(ADI_record), "\t", true_adi/len(test_dataset)*100, "%")
    print(str(PROJ_record), "\t", true_proj/len(test_dataset)*100, "%")
    print(str(RE_record), "\t", true_re/len(test_dataset)*100, "%")
    print(str(TE_record), "\t", true_te/len(test_dataset)*100, "%")
    print("===================================================")  

if __name__ == '__main__':
    main()

