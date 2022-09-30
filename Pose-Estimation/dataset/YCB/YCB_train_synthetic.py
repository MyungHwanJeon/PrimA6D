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

import cv2

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import transform
import view_sampler
from synthetic_dataset_utils import *
import config

from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply

parser = argparse.ArgumentParser(description='dataset')
parser.add_argument("--obj", "-o", required=True, default=1, type=int, help="object number(1 ~ 21)")
parser.add_argument("--path", "-p", required=False, default="../processed_dataset/YCB", help="save path")
args = parser.parse_args()


path = args.path + "/train/obj_" + str(args.obj) + "/synthetic"
os.makedirs(path, exist_ok=True)   
    
def main():

    dataset = synthetic_dataset(dataset="YCB", obj_idx=args.obj) 

    ##########################
    #the path of current file
    current_path = os.path.dirname(os.path.realpath(__file__))
            
    verts, faces, aux = load_obj(current_path + '/../3d_model/YCB/obj_' + str(args.obj) + '/textured_reduced.obj')
    verts = verts * 1000.
    obj_diameter = calc_pts_diameter(verts.data.cpu().numpy().astype(np.float32))

    sampling_size = 50000
    
    views, pts_level = view_sampler.sample_views(sampling_size, mode='hinterstoisser')                
    
    for i in range(sampling_size*2):
        
        print(int(args.obj), ":", i)
        
        if i < sampling_size:
            rot_ = views[i]['R']
            tra_ = np.zeros(3)
        elif i >= sampling_size:
            rot_ = views[int(i-sampling_size)]['R'] @ transform.euler_matrix(180*(np.pi/180.), 0, 0)[:3, :3].astype(np.float32)
            tra_ = np.zeros(3)                    
            
        
        tra_[2] = random.uniform(200, 2500)

        K = np.array([ [1066.778,  0.0,       312.9869], 
                        [0.0,       1067.487,  241.3109], 
                        [0.0,       0.0,        1.0]])
        K = np.array(K).reshape(3,3)

        projection_diameter_x = obj_diameter * K[0, 0] / tra_[2]
        projection_diameter_y = obj_diameter * K[1, 1] / tra_[2]
        
        projection_diameter = np.maximum(projection_diameter_y, projection_diameter_x)
        obj_center_x_min = 0 * (640-projection_diameter) + projection_diameter/2.
        translation_x_min = tra_[2] * (obj_center_x_min - (640/2.))/ K[0,0]  
        obj_center_x_max = (640-projection_diameter) + projection_diameter/2.
        translation_x_max = tra_[2] * (obj_center_x_max - (640/2.))/ K[0,0] 
        
        tra_[0] = random.uniform(translation_x_min, translation_x_max) 
             
        obj_center_y_min = 0 * (480-projection_diameter) + projection_diameter/2.
        translation_y_min = tra_[2] * (obj_center_y_min - (480/2.))/ K[1, 1]
        obj_center_y_max = 1 * (480-projection_diameter) + projection_diameter/2.
        translation_y_max = tra_[2] * (obj_center_y_max - (480/2.))/ K[1, 1]
        
        tra_[1] = random.uniform(translation_y_min, translation_y_max) 

        if translation_x_min > translation_x_max:
            continue
        if translation_y_min > translation_y_max:
            continue
        
        if tra_[0] < translation_x_min:
            tra_[0] = translation_x_min
        if tra_[0] > translation_x_max:   
            tra_[0] = translation_x_max        
           
        state, obj_color, obj_depth, obj_mask, gt_color, primitive_color, primitive_x_color, primitive_y_color, primitive_z_color = dataset.render_synthetic(rot_, tra_)   
                
        if state is True:
        
            dataset_dict = {"obj_inp":np.array(obj_color, dtype=np.uint8),
                            #"obj_depth":np.array(obj_depth, dtype=np.float32),
                            "obj_mask":np.array(obj_mask.astype(np.uint8)*255, dtype=np.uint8),
                            "obj_gt":np.array(gt_color, dtype=np.uint8),
                            "primitive_gt":np.array(primitive_color, dtype=np.uint8),
                            "primitive_x_gt":np.array(primitive_x_color, dtype=np.uint8),                            
                            "primitive_y_gt":np.array(primitive_y_color, dtype=np.uint8),                            
                            "primitive_z_gt":np.array(primitive_z_color, dtype=np.uint8),                            
                            "rot_gt":np.array(rot_, dtype=np.float32),
                            "tra_gt":np.array(tra_, dtype=np.float32),
                            "k":np.array(K, dtype=np.float32)
                            }  
                     
                      
            
            f = gzip.open(path + "/obj_" + str(args.obj) + "_random_" + str(i) + ".gz","wb")
            pickle.dump(dataset_dict,f)
            f.close()  
            
            #cv2.imshow("obj_color", obj_color)
            #cv2.imshow("obj_depth", obj_depth)            
            #cv2.imshow("obj_mask", np.array(obj_mask.astype(np.uint8)*255))
            #cv2.imshow("gt_color", gt_color)
            #cv2.imshow("primitive_color", primitive_color)
            #cv2.imshow("primitive_x_color", primitive_x_color)
            #cv2.imshow("primitive_y_color", primitive_y_color)
            #cv2.imshow("primitive_z_color", primitive_z_color)
            #cv2.waitKey(0)    

if __name__ == '__main__':
    main()        
