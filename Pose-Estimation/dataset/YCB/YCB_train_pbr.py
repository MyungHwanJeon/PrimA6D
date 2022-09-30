import os
import pyglet

import argparse
import numpy as np
import time
import random
import sys
import pathlib
import matplotlib.pyplot as plt
import pickle
import gzip
import imageio
import json

import cv2

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import transform
import view_sampler
from synthetic_dataset_utils import *
import config

from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply


parser = argparse.ArgumentParser(description='dataset')
parser.add_argument("--obj", "-o", required=True, default=1, type=int, help="object number (1 ~ 21)")
parser.add_argument("--path", "-p", required=False, default="../processed_dataset/YCB", help="save path")
args = parser.parse_args()  


path = args.path + "/train/obj_" + str(args.obj) + "/pbr" 
os.makedirs(path, exist_ok=True) 
                     
def make_dataset(obj_id):

    dataset = synthetic_dataset(dataset="YCB", obj_idx=obj_id) 

    ######## Load Data ########               
    rot_data = []
    tra_data = []
    bbox_data = []
    bbox_visib_data = []
    K_data = []
    image_list = []
    depth_list = []
    mask_list = []
    for nn in range(50):
      

        YCB_SCENE_GT_PATH = config.YCBV_dataset_path + "/train_pbr/" + "%06d" % (nn) + "/scene_gt.json"
        YCB_GT_INFO_PATH = config.YCBV_dataset_path + "/train_pbr/" + "%06d" % (nn) + "/scene_gt_info.json"  
        YCB_CAMERA_PATH = config.YCBV_dataset_path + "/train_pbr/" + "%06d" % (nn) + "/scene_camera.json"  

        YCB_camera = open(YCB_CAMERA_PATH)
        YCB_camera = json.load(YCB_camera)         
                    
        YCB_scene_gt = open(YCB_SCENE_GT_PATH)
        YCB_scene_gt = json.load(YCB_scene_gt)

    
        YCB_gt_info = open(YCB_GT_INFO_PATH)
        YCB_gt_info = json.load(YCB_gt_info)
        
        for i in range(len(YCB_scene_gt)):
            for j in range(len(YCB_scene_gt[str(i)])):      
            
                if (YCB_gt_info[str(i)][j]["visib_fract"] < 0.1):
                    continue
            
                if (YCB_gt_info[str(i)][j]["px_count_visib"] == 0):
                    continue
                
                if (YCB_gt_info[str(i)][j]["bbox_visib"][2] <= 0):
                    continue
                    
                if (YCB_gt_info[str(i)][j]["bbox_visib"][3] <= 0):
                    continue  
                  
                if YCB_scene_gt[str(i)][j]["obj_id"] is obj_id:
                    rot_data.append(YCB_scene_gt[str(i)][j]["cam_R_m2c"])
                    tra_data.append(YCB_scene_gt[str(i)][j]["cam_t_m2c"])
                    bbox_data.append(YCB_gt_info[str(i)][j]["bbox_obj"])
                    bbox_visib_data.append(YCB_gt_info[str(i)][j]["bbox_visib"])                                                                                
                    image_list.append(config.YCBV_dataset_path + "/train_pbr/" + "%06d" % (nn) + "/rgb/" + "%06d" % (i) + ".jpg")
                    depth_list.append(config.YCBV_dataset_path + "/train_pbr/" + "%06d" % (nn) + "/depth/" + "%06d" % (i) + ".png")
                    mask_list.append(config.YCBV_dataset_path + "/train_pbr/" + "%06d" % (nn) + "/mask_visib/" + "%06d" % (i) + "_" + "%06d" % (j) + ".png")
                    K_data.append(np.array(YCB_camera[str(i)]["cam_K"]).reshape((3, 3)))
                    
    np_rot_data = np.array(rot_data)
    np_rot_data = np_rot_data.reshape((-1, 3, 3))
    np_tra_data = np.array(tra_data)
    np_bbox_data = np.array(bbox_data)
    np_bbox_visib_data = np.array(bbox_visib_data)  
    np_K_data = np.array(K_data)   

    
    size = np_rot_data.shape[0]
    
    for n in range(size):
        
        print(obj_id, ":", n, "/", size)
        
        if obj_id is 5:
            np_rot_data[n] = np_rot_data[n] @ transform.euler_matrix(0, 0, -23*(np.pi/180.))[:3, :3].astype(np.float32)

        state, obj_color, obj_depth, obj_mask, gt_color, primitive_color, primitive_x_color, primitive_y_color, primitive_z_color = dataset.render_synthetic(np_rot_data[n], np_tra_data[n], np_K_data[n])  

        if state is True:


            obj_color = cv2.imread(image_list[n], cv2.IMREAD_COLOR)               
            
            #obj_depth = imageio.imread(depth_list[n]).astype(np.float32)
            obj_mask = imageio.imread(mask_list[n]).astype(np.bool)
            
            #K = np.array([[1066.778,  0.0,       312.9869], 
            #                [0.0,       1067.487,  241.3109], 
            #                [0.0,       0.0,        1.0]])
            
            #xs, ys = np.meshgrid(np.arange(obj_depth.shape[1]), np.arange(obj_depth.shape[0]))

            #Xs = np.multiply(xs - np_K_data[n, 0, 2], obj_depth) * (1.0 / np_K_data[n, 0, 0])
            #Ys = np.multiply(ys - np_K_data[n, 1, 2], obj_depth) * (1.0 / np_K_data[n, 1, 1])

            #obj_depth = np.sqrt(Xs.astype(np.float32)**2 + Ys.astype(np.float32)**2 + obj_depth.astype(np.float32)**2)
            #obj_depth *= 0.1
          

            dataset_dict = {"obj_inp":np.array(obj_color, dtype=np.uint8),
                            #"obj_depth":np.array(obj_depth, dtype=np.float32),
                            "obj_mask":np.array(obj_mask.astype(np.uint8)*255, dtype=np.uint8),
                            "obj_gt":np.array(gt_color, dtype=np.uint8),
                            "primitive_gt":np.array(primitive_color, dtype=np.uint8),
                            "primitive_x_gt":np.array(primitive_x_color, dtype=np.uint8),                            
                            "primitive_y_gt":np.array(primitive_y_color, dtype=np.uint8),                            
                            "primitive_z_gt":np.array(primitive_z_color, dtype=np.uint8),                           
                            "rot_gt":np.array(np_rot_data[n], dtype=np.float32),
                            "tra_gt":np.array(np_tra_data[n], dtype=np.float32),
                            "k":np.array(np_K_data[n], dtype=np.float32),
                            }                       
                            
            #cv2.imshow("obj_color", obj_color)
            #cv2.imshow("obj_mask", np.array(obj_mask.astype(np.uint8)*255))
            #cv2.imshow("gt_color", gt_color)
            #cv2.imshow("primitive_color", primitive_color)
            #cv2.imshow("primitive_1axis_color", primitive_1axis_color)
            #cv2.waitKey(0)                             

            f = gzip.open(path + "/obj_" + str(args.obj) + "_pbr_" + str(n) + ".gz","wb")
            pickle.dump(dataset_dict,f)
            f.close()
                                

def main():
           
    make_dataset(args.obj)

if __name__ == '__main__':
    main()   
