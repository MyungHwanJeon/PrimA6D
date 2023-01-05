import argparse
import sys
import os
import time
import numpy as np
import cupy as cp
import random
import cv2
from enum import Enum, IntEnum
import atexit
import signal


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import torch
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from trt_utils import utils as trt_utils

from model import utils, PrimA6DPPNet, SegmentationNet
from misc.misc import *

from bop_toolkit_lib.misc import *
import dataset.transform as transform
import trimesh

import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2 as pc2
from sensor_msgs.msg import Image as ros_image
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import Pose
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from std_msgs.msg import Float32
import message_filters
import sensor_msgs.point_cloud2 as pc2

from multiprocessing import Pool, Process, Value, Manager, Lock, shared_memory

parser = argparse.ArgumentParser(
                                description='R6D')
parser.add_argument('-o', '--obj', dest='obj', default="2 3 4 5 6 7 8 9 11 13 15 17 18 20 21",
                    help='object list')                                      
parser.set_defaults(dataset="YCB")#, vis=False)
args = parser.parse_args()  
#print(args)
     
global K
global D   
#K = np.array([1066.778, 0.0, 312.9869, 0.0, 1067.487, 241.3109, 0.0, 0.0, 1.0]).reshape(3, 3)
K = np.eye(3)  
D = np.array([0., 0., 0., 0., 0., 0., 0., 0.])                         

def crop_image_using_bb(img, bb_xywh):

    x, y, w, h = np.array(bb_xywh).astype(np.int32)
    size = int(np.maximum(h, w))
    left = np.maximum(x+w/2-size/2, 0)
    right = np.minimum(x+w/2+size/2, img.shape[1])
    top = np.maximum(y+h/2-size/2, 0)
    bottom = np.minimum(y+h/2+size/2, img.shape[0])

    cropped_img = img[int(top):int(bottom), int(left):int(right)]

    return cropped_img, np.array([int(left), int(top), int(right)-int(left), int(bottom)-int(top)])


def make_padding_bb(img, bb_xywh, pad_factor=1.4):
    
    x, y, w, h = np.array(bb_xywh).astype(np.int32)                               
    size = int(np.maximum(h, w) * pad_factor)   
                
    dimension = [640, 480]
    #dimension = [1920, 1080]

    if x+w/2-size/2 < 0:
        left = 0
        right = np.minimum(x+w/2+size/2, dimension[0])
    elif x+w/2+size/2 > dimension[0]:
        right = dimension[0]
        left = np.maximum(x+w/2-size/2, 0)
    else:
        left = np.maximum(x+w/2-size/2, 0)
        right = np.minimum(x+w/2+size/2, dimension[0])
        
    if y+h/2-size/2 < 0:
        top = 0
        bottom = np.minimum(y+h/2+size/2, dimension[1])
    elif y+h/2+size/2 > dimension[1]:
        bottom = dimension[1]
        top = np.maximum(y+h/2-size/2, 0)
    else:
        top = np.maximum(y+h/2-size/2, 0)
        bottom = np.minimum(y+h/2+size/2, dimension[1])            

    return np.array([int(left), int(top), int(right)-int(left), int(bottom)-int(top)]) 


def calc_2d_bbox(xs, ys, im_size):

    bbTL = (max(xs.min() - 1, 0),
            max(ys.min() - 1, 0))
    bbBR = (min(xs.max() + 1, im_size[0] - 1),
            min(ys.max() + 1, im_size[1] - 1))

    return [bbTL[0], bbTL[1], bbBR[0] - bbTL[0], bbBR[1] - bbTL[1]]      
            

class DATA_FIELD(IntEnum):
    TIME_STAMP = 0 
    IMAGE_COLOR = 1
    IMAGE_DEPTH = 2    
    IMAGE_INPUT = 3
    PUBLISH_FLAG = 4
    CAMERA_INTRINSIC = 5
    CAMERA_DISTORTION = 6   
    
    MODEL_FLAG = 1
    SEGMENTATION = 2
    KEYPOINT = 3
    KEYPOINT_UNC = 4
    UNC_NORM = 5
    BOUNDING_BOX = 6
    ROT = 7
    TRA = 8
    IDX = 9

    RESULT_IMAGE_FLAG = 0
    RESULT_IMAGE = 1
    RESULT_ROT = 2
    RESULT_TRA = 3    
    RESULT_CAMERA_INTRINSIC = 4

class INPUT_DATA():

    def __init__(self):  

        self.shared_datas = []
        self.shms = []
        self.lock = Lock()
        
        ## "time_stamp" ##
        d = np.zeros(shape=(1), dtype=np.float64)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:] = np.ones(shape=(1), dtype=np.float64)
        self.shared_datas.append(shared_data)
        self.shms.append(shm)                       

        ## image color ##
        d = np.zeros(shape=(480, 640, 3), dtype=np.uint8)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:, :, :] = np.zeros(shape=(480, 640, 3), dtype=np.uint8)
        self.shared_datas.append(shared_data)
        self.shms.append(shm)     

        ## image depth ##
        d = np.zeros(shape=(480, 640), dtype=np.uint16)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:, :] = np.zeros(shape=(480, 640), dtype=np.uint16)
        self.shared_datas.append(shared_data)
        self.shms.append(shm)                    
        
        ## image input ##
        d = np.zeros(shape=(1, 3, 480, 640), dtype=np.float32)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:, :, :, :] = np.zeros(shape=(1, 3, 480, 640), dtype=np.float32)
        self.shared_datas.append(shared_data)
        self.shms.append(shm)   

        ## "publish_flag" ##
        d = np.zeros(shape=(1), dtype=np.bool_)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:] = np.zeros(shape=(1), dtype=np.bool_)
        self.shared_datas.append(shared_data)
        self.shms.append(shm)    
        
        ## "camera intrinsic" ##
        d = np.zeros(shape=(3, 3), dtype=np.float64)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:, :] = np.ones(shape=(3, 3), dtype=np.float64)
        self.shared_datas.append(shared_data)
        self.shms.append(shm) 
        
        ## "camera distortion" ##
        d = np.zeros(shape=(8), dtype=np.float64)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:] = np.ones(shape=(8), dtype=np.float64)
        self.shared_datas.append(shared_data)
        self.shms.append(shm)           

    def __del__(self):

        for i in range(len(self.shms)):
            self.shms[i].close()
            self.shms[i].unlink()          
        
    def get_data(self, field):

        self.lock.acquire()        
        field_value = np.ndarray(self.shared_datas[field].shape, dtype=self.shared_datas[field].dtype, buffer=self.shms[field].buf)        
        self.lock.release()

        return field_value     

    def set_data(self, field, data):

        self.lock.acquire() 
        field_value = np.ndarray(self.shared_datas[field].shape, dtype=self.shared_datas[field].dtype, buffer=self.shms[field].buf)

        if len(data.shape) == 1:
            field_value[:] = data
        elif len(data.shape) == 2:
            field_value[:, :] = data
        elif len(data.shape) == 3:
            field_value[:, :, :] = data
        elif len(data.shape) == 4:
            field_value[:, :, :, :] = data                        
        
        self.lock.release()                        
                      

class RESULT_IMAGE_DATA():

    def __init__(self, obj_idxs):  

        self.obj_idxs = obj_idxs

        self.shared_datas = dict()
        self.shms = dict()
        self.lock = Lock()

        ## "result_iamge_flag" ##
        d = np.zeros(shape=(1), dtype=np.bool_)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:] = np.zeros(shape=(1), dtype=np.bool_)
        self.shared_datas[DATA_FIELD.RESULT_IMAGE_FLAG] = shared_data
        self.shms[DATA_FIELD.RESULT_IMAGE_FLAG] = shm        

        ## result_iamge ##
        d = np.zeros(shape=(480, 640, 3), dtype=np.uint8)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:, :, :] = np.zeros(shape=(480, 640, 3), dtype=np.uint8)
        self.shared_datas[DATA_FIELD.RESULT_IMAGE] = shared_data
        self.shms[DATA_FIELD.RESULT_IMAGE] = shm 
        
        ## "result Rot" ##
        for i in self.obj_idxs:
            d = np.zeros(shape=(3, 3), dtype=np.float32)
            shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
            shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
            shared_data[:, :] = np.zeros(shape=(3, 3), dtype=np.float32)
            self.shared_datas[DATA_FIELD.RESULT_ROT+(i*0.01)] = shared_data
            self.shms[DATA_FIELD.RESULT_ROT+(i*0.01)] = shm

        ## "t" ##
        for i in self.obj_idxs:
            d = np.zeros(shape=(3), dtype=np.float32)
            shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
            shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
            shared_data[:] = np.zeros(shape=(3), dtype=np.float32)
            self.shared_datas[DATA_FIELD.RESULT_TRA+(i*0.01)] = shared_data
            self.shms[DATA_FIELD.RESULT_TRA+(i*0.01)] = shm
            
        ## "result camera intrinsic" ##
        d = np.zeros(shape=(3, 3), dtype=np.float64)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:, :] = np.ones(shape=(3, 3), dtype=np.float64)
        self.shared_datas[DATA_FIELD.RESULT_CAMERA_INTRINSIC] = shared_data
        self.shms[DATA_FIELD.RESULT_CAMERA_INTRINSIC] = shm            


    def __del__(self):

        
        self.shms[DATA_FIELD.RESULT_IMAGE_FLAG].close()
        self.shms[DATA_FIELD.RESULT_IMAGE_FLAG].unlink()

        self.shms[DATA_FIELD.RESULT_CAMERA_INTRINSIC].close()
        self.shms[DATA_FIELD.RESULT_CAMERA_INTRINSIC].unlink()  
                     
        self.shms[DATA_FIELD.RESULT_IMAGE].close()
        self.shms[DATA_FIELD.RESULT_IMAGE].unlink()  

        for i in self.obj_idxs:
            self.shms[DATA_FIELD.RESULT_ROT+(i*0.01)].close()
            self.shms[DATA_FIELD.RESULT_ROT+(i*0.01)].unlink()                                

            self.shms[DATA_FIELD.RESULT_TRA+(i*0.01)].close()
            self.shms[DATA_FIELD.RESULT_TRA+(i*0.01)].unlink()                                

    def get_data(self, field, idx=0):

        self.lock.acquire()        
        field_value = np.ndarray(self.shared_datas[field+idx*0.01].shape, dtype=self.shared_datas[field+idx*0.01].dtype, buffer=self.shms[field+idx*0.01].buf)        
        self.lock.release()

        return field_value     

    def set_data(self, field, data, idx=0):

        self.lock.acquire() 
        field_value = np.ndarray(self.shared_datas[field+idx*0.01].shape, dtype=self.shared_datas[field+idx*0.01].dtype, buffer=self.shms[field+idx*0.01].buf)
        
        if len(data.shape) == 1:
            field_value[:] = data
        elif len(data.shape) == 2:
            field_value[:, :] = data
        elif len(data.shape) == 3:
            field_value[:, :, :] = data
        elif len(data.shape) == 4:
            field_value[:, :, :, :] = data  

        self.lock.release()    

    
class DETECTION_DATA():

    def __init__(self):  

        self.shared_datas = []
        self.shms = []
        self.lock = Lock()

        ## "time_stamp" ##
        d = np.zeros(shape=(1), dtype=np.float64)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:] = np.ones(shape=(1), dtype=np.float64)
        self.shared_datas.append(shared_data)
        self.shms.append(shm) 

        ## "model_flag" ##
        d = np.zeros(shape=(1), dtype=np.bool_)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:] = np.ones(shape=(1), dtype=np.bool_)
        self.shared_datas.append(shared_data)
        self.shms.append(shm)        

        ## "segmentation" ##
        d = np.zeros(shape=(480, 640, 1), dtype=np.uint8)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:, :, :] = np.zeros(shape=(480, 640, 1), dtype=np.uint8)
        self.shared_datas.append(shared_data)
        self.shms.append(shm)              
        
        ## "keypoint" ##
        d = np.zeros(shape=(42, 2), dtype=np.float32)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:, :] = np.zeros(shape=(42, 2), dtype=np.float32)
        self.shared_datas.append(shared_data)
        self.shms.append(shm)         
    
        ## "keypoint unc" ##
        d = np.zeros(shape=(3), dtype=np.float32)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:] = np.zeros(shape=(3), dtype=np.float32)
        self.shared_datas.append(shared_data)
        self.shms.append(shm)         
        
        ## "unc_norm" ##
        d = np.zeros(shape=(1), dtype=np.float32)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:] = np.zeros(shape=(1), dtype=np.float32)
        self.shared_datas.append(shared_data)
        self.shms.append(shm)         
        
        ## "bounding_box" ##
        d = np.zeros(shape=(4), dtype=np.float32)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:] = np.zeros(shape=(4), dtype=np.float32)
        self.shared_datas.append(shared_data)
        self.shms.append(shm)         
        
        ## "R" ##
        d = np.zeros(shape=(3, 3), dtype=np.float32)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:, :] = np.zeros(shape=(3, 3), dtype=np.float32)
        self.shared_datas.append(shared_data)
        self.shms.append(shm)         
        
        ## "t" ##
        d = np.zeros(shape=(3), dtype=np.float32)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:] = np.zeros(shape=(3), dtype=np.float32)
        self.shared_datas.append(shared_data)
        self.shms.append(shm)        
        
        ## "idx" ##
        d = np.zeros(shape=(1), dtype=np.int32)
        shm = shared_memory.SharedMemory(create=True, size=d.nbytes)
        shared_data = np.ndarray(d.shape, dtype=d.dtype, buffer=shm.buf)
        shared_data[:] = np.zeros(shape=(1), dtype=np.int32)
        self.shared_datas.append(shared_data)
        self.shms.append(shm)    

    def __del__(self):

        for i in range(len(self.shms)):
            self.shms[i].close()
            self.shms[i].unlink()
                     

    def get_data(self, field):

        self.lock.acquire()        
        field_value = np.ndarray(self.shared_datas[field].shape, dtype=self.shared_datas[field].dtype, buffer=self.shms[field].buf)        
        self.lock.release()

        return field_value

    def set_data(self, field, data):

        self.lock.acquire() 
        field_value = np.ndarray(self.shared_datas[field].shape, dtype=self.shared_datas[field].dtype, buffer=self.shms[field].buf)
        
        if len(data.shape) == 1:
            field_value[:] = data
        elif len(data.shape) == 2:
            field_value[:, :] = data
        elif len(data.shape) == 3:
            field_value[:, :, :] = data
        elif len(data.shape) == 4:
            field_value[:, :, :, :] = data  

        self.lock.release()                 


def thread_inference(obj_idx, gpu_idx, input_data, detection_data):

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_idx)
        
    segmentation_model = SegmentationNet.SegmentationNet(device=torch.device("cuda:"+str(gpu_idx)))
    segmentation_model.to(torch.device("cuda:"+str(gpu_idx)))       
    segmentation_model.eval()        
    ret = load_weight_all(segmentation_model, pre_trained_path="../Segmentation/trained_weight/obj_" + str(obj_idx) + "_S.pth", device=torch.device("cuda:"+str(gpu_idx)))  
    if ret is False:
        return 
    #print("load torch32 Segmentator of obj", obj_idx)
    
    sym_obj = [13, 16, 19, 20, 21]
    
    prima6d_model = PrimA6DPPNet.PrimA6DNet(device=torch.device("cuda:"+str(gpu_idx)))
    prima6d_model.to(torch.device("cuda:"+str(gpu_idx)))
    prima6d_model.eval()       
    pre_trained_path="../PrimA6D++/trained_weight/obj_" + str(obj_idx) + "_P.pth"#_add_best"
    if obj_idx in sym_obj:
        pre_trained_path = "../PrimA6D++/trained_weight/obj_" + str(obj_idx) + "_P.pth_adi_best"        
    ret = load_weight_all(prima6d_model, pre_trained_path=pre_trained_path, device=torch.device("cuda:"+str(gpu_idx)))
    if ret is False:
        return
    #print("load torch32 PrimA6D weight of obj", obj_idx)                         
    
    obj_model_path = "../dataset/3d_model/YCB/model_eval/obj_" +  "%06d" % (obj_idx) + '.ply'       
    obj_model = trimesh.load(obj_model_path)                    
    model_diameter = calc_pts_diameter2(obj_model.vertices) 

    primitive_model_path = '../dataset/3d_model/PRIMITIVE/1axis.ply'   
    primitive_model = trimesh.load(primitive_model_path)      
    primitive_diameter = calc_pts_diameter2(primitive_model.vertices)
    primitive_scale = model_diameter*0.5 / primitive_diameter  

    kp_3d_4d = np.array([                           
                        [[0, 0, 0, 1], [15, 15, 15, 1], [15, 15, -15, 1], [15, -15, 15, 1], [15, -15, -15, 1], [-15, 15, 15, 1], [-15, 15, -15, 1], [-15, -15, 15, 1], [-15, -15, -15, 1],
                        [105, 0, 0, 1], [105, 15, 15, 1], [105, 15, -15, 1], [105, -15, 15, 1], [105, -15, -15, 1]],                  
                        
                        [[0, 0, 0, 1], [15, 15, 15, 1], [15, 15, -15, 1], [15, -15, 15, 1], [15, -15, -15, 1], [-15, 15, 15, 1], [-15, 15, -15, 1], [-15, -15, 15, 1], [-15, -15, -15, 1],
                        [0, 105, 0, 1], [15, 105, 15, 1], [15, 105, -15, 1], [-15, 105, 15, 1], [-15, 105, -15, 1]],                    
                        
                        [[0, 0, 0, 1], [15, 15, 15, 1], [15, 15, -15, 1], [15, -15, 15, 1], [15, -15, -15, 1], [-15, 15, 15, 1], [-15, 15, -15, 1], [-15, -15, 15, 1], [-15, -15, -15, 1],
                        [0, 0, 105, 1], [15, 15, 105, 1], [15, -15, 105, 1], [-15, 15, 105, 1], [-15, -15, 105, 1]],                                        
                        ], dtype=np.float32)               
    kp_3d_4d[:, :, 0:3] = kp_3d_4d[:, :, 0:3] * primitive_scale 
    kp_3d_4d = kp_3d_4d.reshape((14*3, 4))
    
    print("[Thread Inference] : Start - obj:" + str(obj_idx) + ", cuda:" + str(gpu_idx))
   
    while(1):
        try:                 

            if detection_data.get_data(DATA_FIELD.MODEL_FLAG)[0] == True:                                     
                                
                stamp = input_data.get_data(DATA_FIELD.TIME_STAMP)

                segmentation_inp = input_data.get_data(DATA_FIELD.IMAGE_INPUT)
                segmentation_inp = torch.from_numpy(segmentation_inp).to(torch.device("cuda:"+str(gpu_idx))).float()               
                                                   
                segmentation_pred = segmentation_model(segmentation_inp)
                segmentation_pred = torch.argmax(segmentation_pred, dim=1)                                                                    
                
                obj_mask_idx = torch.nonzero(segmentation_pred.view(480, 640))                

                if obj_mask_idx.shape[0] != 0:                      

                    obj_bb_xywh = calc_2d_bbox(obj_mask_idx[:, 1], obj_mask_idx[:, 0], [640, 480, 3])

                    x = obj_bb_xywh[0]#*3
                    y = obj_bb_xywh[1]#*2.25
                    w = obj_bb_xywh[2]#*3
                    h = obj_bb_xywh[3]#*2.25
                    
                    #x = obj_bb_xywh[0]*2.25+240
                    #y = obj_bb_xywh[1]*2.25
                    #w = obj_bb_xywh[2]*2.25
                    #h = obj_bb_xywh[3]*2.25

                    obj_bb_xywh = np.array([x, y, w, h])   

                    crop_img, obj_bb_square_xywh = crop_image_using_bb(img=input_data.get_data(DATA_FIELD.IMAGE_COLOR), bb_xywh=obj_bb_xywh)                
                    padding_obj_bb_xywh = make_padding_bb(img=crop_img, bb_xywh=obj_bb_square_xywh, pad_factor=1.4)
                    
                    bb_size = int(np.maximum(padding_obj_bb_xywh[2], padding_obj_bb_xywh[3]))
                    left = np.maximum(obj_bb_square_xywh[0] - padding_obj_bb_xywh[0], 0)
                    right = np.minimum(left + obj_bb_square_xywh[2], bb_size)
                    top = np.maximum(obj_bb_square_xywh[1] - padding_obj_bb_xywh[1], 0)
                    bottom = np.minimum(top + obj_bb_square_xywh[3], bb_size)
                    crop_img = cv2.resize(crop_img, (int(right)-int(left), int(bottom)-int(top)), interpolation=cv2.INTER_NEAREST) 
                    blank_image = np.zeros((int(bb_size), int(bb_size), 3), dtype=np.uint8) 
                    blank_image[int(top):int(bottom), int(left):int(right)] = crop_img           
                    blank_image = blank_image[0:padding_obj_bb_xywh[3], 0:padding_obj_bb_xywh[2]]        

                    result_image_size = (64, 64)
                    if padding_obj_bb_xywh[2] > result_image_size[0] or padding_obj_bb_xywh[3] > result_image_size[0]:
                        blank_image = cv2.resize(blank_image, result_image_size, interpolation=cv2.INTER_AREA)
                    else:
                        blank_image = cv2.resize(blank_image, result_image_size, interpolation=cv2.INTER_LINEAR_EXACT)   
                   
                    PrimA6D_inp = torch.from_numpy(blank_image).to(torch.device("cuda:"+str(gpu_idx))).float() / 255.              
                    PrimA6D_inp = PrimA6D_inp.permute(2, 0, 1).view(1, 3, 64, 64)           

                    results = prima6d_model(PrimA6D_inp)
                    primitive_out = results[0].data.cpu().numpy()[0]
                    keypoint_out = results[1].data.cpu().numpy()[0]
                    uncertainty_out = results[2].data.cpu().numpy()[0]
                                        
                    kp_out = keypoint_out.reshape((14*3, 2))
                    
                    conf_threshold = 0.4
                    valid_cnt = 0
                    valid_axis = np.array([False, False, False], dtype=np.bool_)
                    
                    confidence_norm = np.array([0, 0, 0], dtype=np.float32)
                    
                    if np.exp(uncertainty_out[0]) < conf_threshold:
                        valid_cnt += 1
                        valid_axis[0] = True
                        confidence_norm[0] = np.exp(uncertainty_out[0])
                    if np.exp(uncertainty_out[1]) < conf_threshold:
                        valid_cnt += 1
                        valid_axis[1] = True
                        confidence_norm[1] = np.exp(uncertainty_out[1])
                    if np.exp(uncertainty_out[2]) < conf_threshold:
                        valid_cnt += 1
                        valid_axis[2] = True
                        confidence_norm[2] = np.exp(uncertainty_out[2])
                        
                    if valid_cnt == 0:
                        confidence_norm = np.exp(uncertainty_out)
                    
                    confidence_norm = np.linalg.norm(confidence_norm)
                    
                    t_est = np.zeros((3), dtype=np.float32)
                    R_est = np.eye(3, dtype=np.float32)
                    
                    if valid_cnt > 0:                        

                        kp_3d_pts = np.zeros((9 + valid_cnt * 5, 4), dtype=np.float32)
                        kp_2d_pts = np.zeros((9 + valid_cnt * 5, 2), dtype=np.float32)
                        
                        start_idx = 9                            
                        if valid_axis[0]:
                        
                            for iii in range(9):
                                kp_3d_pts[iii] = kp_3d_4d[iii]
                                kp_2d_pts[iii] += kp_out[iii]
                            
                            for iii in range(5):
                                kp_3d_pts[start_idx + iii] = kp_3d_4d[0 + 9 + iii]
                                kp_2d_pts[start_idx + iii] = kp_out[0 + 9 + iii]
                            start_idx += 5    
                                
                        if valid_axis[1]:
                        
                            for iii in range(9):
                                kp_3d_pts[iii] = kp_3d_4d[iii]
                                kp_2d_pts[iii] += kp_out[14 + iii]
                                    
                            for iii in range(5):
                                kp_3d_pts[start_idx + iii] = kp_3d_4d[14 + 9 + iii]
                                kp_2d_pts[start_idx + iii] = kp_out[14 + 9 + iii]
                            start_idx += 5
                                
                        if valid_axis[2]:
                        
                            for iii in range(9):
                                kp_3d_pts[iii] = kp_3d_4d[iii]
                                kp_2d_pts[iii] += kp_out[28 + iii]
                            
                            for iii in range(5):
                                kp_3d_pts[start_idx + iii] = kp_3d_4d[28 + 9 + iii]
                                kp_2d_pts[start_idx + iii] = kp_out[28 + 9 + iii]  
                            start_idx += 5      
                            
                        for iii in range(9):
                            kp_2d_pts[iii] /= valid_cnt   
                            
                        kp_2d_pts[:, 0] = kp_2d_pts[:, 0] * ((padding_obj_bb_xywh[2])/64) + padding_obj_bb_xywh[0]
                        kp_2d_pts[:, 1] = kp_2d_pts[:, 1] * ((padding_obj_bb_xywh[3])/64) + padding_obj_bb_xywh[1]                    
                            
                            
                        retval, rvec, t_est = cv2.solvePnP(
                                                objectPoints = np.ascontiguousarray(kp_3d_pts[:, 0:3].reshape((-1,1,3)).astype(np.float32)), 
                                                imagePoints = np.ascontiguousarray(kp_2d_pts.reshape((-1,1,2)).astype(np.float32)), 
                                                cameraMatrix = input_data.get_data(DATA_FIELD.CAMERA_INTRINSIC),
                                                distCoeffs = input_data.get_data(DATA_FIELD.CAMERA_DISTORTION),
                                                flags=cv2.SOLVEPNP_ITERATIVE
                                                )                                                                                                                                                                
                        R_est, _ = cv2.Rodrigues(rvec) 
                        
                        
                        detection_data.set_data(DATA_FIELD.TIME_STAMP, stamp)
                        detection_data.set_data(DATA_FIELD.SEGMENTATION, (segmentation_pred.data.cpu().numpy()).reshape((480, 640, 1)))
                        detection_data.set_data(DATA_FIELD.KEYPOINT, kp_out)
                        detection_data.set_data(DATA_FIELD.KEYPOINT_UNC, np.exp(uncertainty_out))
                        detection_data.set_data(DATA_FIELD.UNC_NORM, np.array([confidence_norm], dtype=np.float32))
                        detection_data.set_data(DATA_FIELD.BOUNDING_BOX, obj_bb_xywh)
                        detection_data.set_data(DATA_FIELD.ROT, R_est)
                        detection_data.set_data(DATA_FIELD.TRA, t_est.reshape(3))
                        detection_data.set_data(DATA_FIELD.MODEL_FLAG, np.array([False], dtype=np.bool_))
                        detection_data.set_data(DATA_FIELD.IDX, np.array([obj_idx], dtype=np.int32))                                           
                                                                                   
                    else:
                        detection_data.set_data(DATA_FIELD.MODEL_FLAG, np.array([False], dtype=np.bool_))
                        detection_data.set_data(DATA_FIELD.IDX, np.array([0], dtype=np.int32))                            
                        
                else:
                    detection_data.set_data(DATA_FIELD.MODEL_FLAG, np.array([False], dtype=np.bool_))
                    detection_data.set_data(DATA_FIELD.IDX, np.array([0], dtype=np.int32))                      

            #time.sleep(0.0001)                                    
                            
        except (KeyboardInterrupt, SystemExit):            
            return            

def rospy_shutdown(signal, frame):
    rospy.signal_shutdown("shut down")
    sys.exit(0)

def thread_result_image(result_image_data, obj_idxs):

    signal.signal(signal.SIGINT, rospy_shutdown)
    # ros init #
    rospy.init_node('PrimA6D', anonymous=True) 
    ros_result_image_pub = rospy.Publisher("/pose_estimation/PrimA6D/result_img", ros_image, queue_size=1)

    obj_models = dict()
    for i in obj_idxs:
        obj_model_path = "../dataset/3d_model/YCB/model_eval/obj_" +  "%06d" % (i) + '.ply'       
        obj_models[i] = trimesh.load(obj_model_path)

    obj_colors = dict()
    obj_colors[1] = [0, 0, 0]
    obj_colors[2] = [255, 0, 0]
    obj_colors[3] = [0, 255, 0]
    obj_colors[4] = [0, 0, 255]
    obj_colors[5] = [255, 255, 0]
    obj_colors[6] = [255, 0, 255]
    obj_colors[7] = [0, 255, 255]
    obj_colors[8] = [255, 255, 255]
    obj_colors[9] = [128, 0, 0]
    obj_colors[10] = [0, 128, 0]
    obj_colors[11] = [0, 0, 128]
    obj_colors[12] = [128, 128, 0]
    obj_colors[13] = [128, 0, 128]
    obj_colors[14] = [0, 128, 128]
    obj_colors[15] = [128, 128, 128]
    obj_colors[16] = [255, 128, 0]
    obj_colors[17] = [255, 0, 128]
    obj_colors[18] = [128, 255, 0]
    obj_colors[19] = [0, 255, 128]
    obj_colors[20] = [128, 0, 255]
    obj_colors[21] = [0, 128, 255]

    print("[Thread Result Image] : Start")

    while(1):
        try:             

            if result_image_data.get_data(DATA_FIELD.RESULT_IMAGE_FLAG)[0] == True:

                result_camera_color = result_image_data.get_data(DATA_FIELD.IMAGE_COLOR)                                                     

                for i in obj_idxs:

                    rot = result_image_data.get_data(DATA_FIELD.RESULT_ROT, i).reshape((3, 3))
                    tra = result_image_data.get_data(DATA_FIELD.RESULT_TRA, i).reshape((3, 1))

                    if (tra[0] == -1 and tra[1] == -1 and tra[2] == -1):
                       continue

                    minx = np.min(obj_models[i].vertices[:,0])
                    miny = np.min(obj_models[i].vertices[:,1])
                    minz = np.min(obj_models[i].vertices[:,2])
                    sizex = np.max(obj_models[i].vertices[:,0]) - np.min(obj_models[i].vertices[:,0])
                    sizey = np.max(obj_models[i].vertices[:,1]) - np.min(obj_models[i].vertices[:,1])
                    sizez = np.max(obj_models[i].vertices[:,2]) - np.min(obj_models[i].vertices[:,2])

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

                    proj_est = project_pts(corner_pt, result_image_data.get_data(DATA_FIELD.RESULT_CAMERA_INTRINSIC), rot, tra)        
                    pts = np.array([proj_est[0], proj_est[1], proj_est[3], proj_est[2], proj_est[0], proj_est[4], proj_est[6], proj_est[2]], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(result_camera_color, [pts], True, obj_colors[i], 2)            
                    pts = np.array([proj_est[5], proj_est[4], proj_est[6], proj_est[7], proj_est[5], proj_est[1], proj_est[3], proj_est[7]], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(result_camera_color, [pts], True, obj_colors[i], 2)

                ros_result_image_pub.publish(ros_numpy.msgify(ros_image, result_camera_color, encoding='bgr8')) 
                result_image_data.set_data(DATA_FIELD.RESULT_IMAGE_FLAG, np.array([False], dtype=np.bool_))

            #time.sleep(0.0001)

        except (KeyboardInterrupt, SystemExit):                        
            return   


def thread_publish(input_data, detection_data, obj_idxs, result_image_data):

    signal.signal(signal.SIGINT, rospy_shutdown)
    # ros init #
    rospy.init_node('PrimA6D', anonymous=True)                                     
    ros_camera_info_pub = rospy.Publisher("/pose_estimation/PrimA6D/camera_info", CameraInfo, queue_size=1)
    ros_camera_color_pub = rospy.Publisher("/pose_estimation/PrimA6D/color_raw", ros_image, queue_size=1) 
    ros_camera_depth_pub = rospy.Publisher("/pose_estimation/PrimA6D/depth_raw", ros_image, queue_size=1)     
    ros_detection_result_pub = rospy.Publisher("/pose_estimation/PrimA6D/detection2D_array", Detection2DArray, queue_size=1)

    prev_time_stamp = 0

    print("[Thread Publish] : Start")

    while(1):
        try: 

            publish_flag = True
            for i in obj_idxs:                                                    
                if detection_data[i].get_data(DATA_FIELD.IDX)[0] == -1:
                    publish_flag = False
                    break

            if publish_flag == True:
                
                current_camera_intrinsic = input_data.get_data(DATA_FIELD.CAMERA_INTRINSIC)
                current_camera_distortion = input_data.get_data(DATA_FIELD.CAMERA_DISTORTION)
                current_camera_color = input_data.get_data(DATA_FIELD.IMAGE_COLOR)
                current_camera_depth = input_data.get_data(DATA_FIELD.IMAGE_DEPTH) 
                current_stamp = input_data.get_data(DATA_FIELD.TIME_STAMP)
                
                camera_info_msg = CameraInfo()
                camera_info_msg.header.stamp = rospy.Time.from_sec(current_stamp[0])
                camera_info_msg.D = current_camera_distortion.flatten().tolist()[:5]
                camera_info_msg.K = current_camera_intrinsic.flatten().tolist()
                camera_info_msg.width = current_camera_color.shape[1]
                camera_info_msg.height = current_camera_color.shape[0]   

                current_camera_color_ = cv2.cvtColor(current_camera_color, cv2.COLOR_RGB2BGR)
                camera_color_msg = ros_numpy.msgify(ros_image, current_camera_color_, encoding='bgr8')
                camera_color_msg.header.stamp = rospy.Time.from_sec(current_stamp[0])
                camera_depth_msg = ros_numpy.msgify(ros_image, current_camera_depth, encoding='mono16')
                camera_depth_msg.header.stamp = rospy.Time.from_sec(current_stamp[0])                                
                    
                detection_array_msg = Detection2DArray()
                detection_array_msg.header.stamp = rospy.Time.from_sec(current_stamp[0])

                if prev_time_stamp != current_stamp:                
                    for i in obj_idxs:                            

                        if detection_data[i].get_data(DATA_FIELD.IDX)[0] == i and detection_data[i].get_data(DATA_FIELD.TIME_STAMP)[0] == current_stamp:

                            segmentation = detection_data[i].get_data(DATA_FIELD.SEGMENTATION)
                            keypoint = detection_data[i].get_data(DATA_FIELD.KEYPOINT)
                            keypoint_unc = detection_data[i].get_data(DATA_FIELD.KEYPOINT_UNC)
                            unc_norm = detection_data[i].get_data(DATA_FIELD.UNC_NORM)
                            bounding_box = detection_data[i].get_data(DATA_FIELD.BOUNDING_BOX)
                            rot = detection_data[i].get_data(DATA_FIELD.ROT)
                            tra = detection_data[i].get_data(DATA_FIELD.TRA)                        
                                                                                    
                            detection_msg = Detection2D()
                            detection_msg.header.stamp = rospy.Time.from_sec(current_stamp[0])
                            detection_msg.header.frame_id = "YCB"
                            detection_msg.bbox.center.x = bounding_box[0] + (bounding_box[2] / 2)
                            detection_msg.bbox.center.y = bounding_box[1] + (bounding_box[3] / 2)
                            detection_msg.bbox.center.theta = 0
                            detection_msg.bbox.size_x = bounding_box[2]
                            detection_msg.bbox.size_y = bounding_box[3]
                                                    
                            hypothesis = ObjectHypothesisWithPose()
                            hypothesis.id = i
                            hypothesis.score = unc_norm[0]
                            hypothesis.pose.pose.position.x = tra[0]
                            hypothesis.pose.pose.position.y = tra[1]
                            hypothesis.pose.pose.position.z = tra[2]
                            quat = transform.quaternion_from_matrix(rot[:])
                            hypothesis.pose.pose.orientation.w = quat[0]
                            hypothesis.pose.pose.orientation.x = quat[1]
                            hypothesis.pose.pose.orientation.y = quat[2]
                            hypothesis.pose.pose.orientation.z = quat[3]
                            covariance = np.zeros((6, 6), dtype=np.float64)
                            covariance[0, 0] = keypoint_unc[0]
                            covariance[1, 1] = keypoint_unc[1]
                            covariance[2, 2] = keypoint_unc[2]
                            hypothesis.pose.covariance = covariance.flatten().tolist()
                            detection_msg.results.append(hypothesis)                
                            detection_msg.source_img = ros_numpy.msgify(ros_image, segmentation, encoding='mono8')                
                            detection_array_msg.detections.append(detection_msg)   
                                                        
                            result_image_data.set_data(DATA_FIELD.RESULT_ROT, rot[:], idx=i)
                            result_image_data.set_data(DATA_FIELD.RESULT_TRA, tra[:], idx=i)

                        else:
                            result_image_data.set_data(DATA_FIELD.RESULT_ROT, np.eye(3, dtype=np.float32), idx=i)
                            result_image_data.set_data(DATA_FIELD.RESULT_TRA, np.array(([-1,-1,-1]), dtype=np.float32), idx=i) 
                                                
                    result_image_data.set_data(DATA_FIELD.RESULT_IMAGE, current_camera_color)        
                    result_image_data.set_data(DATA_FIELD.RESULT_IMAGE_FLAG, np.array([True], dtype=np.bool_))

                    ros_detection_result_pub.publish(detection_array_msg)   
                    ros_camera_info_pub.publish(camera_info_msg)                                   
                    ros_camera_color_pub.publish(camera_color_msg)
                    ros_camera_depth_pub.publish(camera_depth_msg)

                prev_time_stamp = current_stamp.copy()

                for i in obj_idxs:                    
                    detection_data[i].set_data(DATA_FIELD.IDX, np.array([-1], dtype=np.int32))  
                                
                input_data.set_data(DATA_FIELD.PUBLISH_FLAG, np.array([False], dtype=np.bool_))

            #time.sleep(0.0001)

        except (KeyboardInterrupt, SystemExit):                        
            return               

class PRIMA6D_ROS():

    def __init__(self):   
        
        if len(args.obj) < 1:
            return
            
        splited_obj = args.obj.split(" ")
        if len(splited_obj) < 1:
            return
            
        self.obj_idxs = []    
        for i in range(len(splited_obj)):
            self.obj_idxs.append(int(splited_obj[i]))
                         
        n_gpu = torch.cuda.device_count()
        gpu_split = np.array_split(self.obj_idxs, n_gpu)

        
        # data definition #               
        self.input_data = INPUT_DATA()
        self.result_image_data = RESULT_IMAGE_DATA(self.obj_idxs)
        self.detection_data = dict()        
        for i in self.obj_idxs:
            self.detection_data[i] = DETECTION_DATA()

        self.new_image_call_cnt = 0
            
        rospy.init_node('PrimA6D', anonymous=True)               
        self.ros_camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.callback_camera_info, queue_size=1)         
        self.ros_image_sub = message_filters.Subscriber('/camera/color/image_raw', ros_image)
        self.ros_depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', ros_image)
        self.ts = message_filters.TimeSynchronizer([self.ros_image_sub, self.ros_depth_sub], 10)
        self.ts.registerCallback(self.callback)           
                  
        self.procs = []        
        for i in range(n_gpu):
            for j in range(gpu_split[i].shape[0]):                
                self.procs.append(Process(target=thread_inference, args=(gpu_split[i][j], i, self.input_data, self.detection_data[gpu_split[i][j]])))
        self.procs.append(Process(target=thread_publish, args=(self.input_data, self.detection_data, self.obj_idxs, self.result_image_data)))
        self.procs.append(Process(target=thread_result_image, args=(self.result_image_data, self.obj_idxs)))

        for p in self.procs:
            p.start()              

    def __del__(self):

        if 'self.procs' in locals():
            for p in self.procs:
                p.terminate() 
          
    def callback_camera_info(self, camera_info):
    
        K = np.array(camera_info.K, dtype=np.float64).reshape(3, 3) 
        D = np.array([camera_info.D[0], camera_info.D[1], camera_info.D[2], camera_info.D[3], camera_info.D[4], 0., 0., 0.], dtype=np.float64)
        
        self.input_data.set_data(DATA_FIELD.CAMERA_INTRINSIC, K)
        self.result_image_data.set_data(DATA_FIELD.RESULT_CAMERA_INTRINSIC, K)
        self.input_data.set_data(DATA_FIELD.CAMERA_DISTORTION, D)

        self.ros_camera_info_sub.unregister()
            
          
    def callback(self, color, depth):

        self.new_image_call_cnt += 1
        if (self.new_image_call_cnt > 2 and self.input_data.get_data(DATA_FIELD.PUBLISH_FLAG)[0] == True):            
            self.input_data.set_data(DATA_FIELD.PUBLISH_FLAG, np.array([False], dtype=np.bool_))
        
        if self.input_data.get_data(DATA_FIELD.PUBLISH_FLAG)[0] == False:  

            self.new_image_call_cnt = 0          
            
            stamp = color.header.stamp.to_sec()        

            img_color = ros_numpy.numpify(color)   
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

            #img_input = cv2.resize(img_color, (640, 480), interpolation=cv2.INTER_AREA) 
            img_input = img_color.reshape((1, 480, 640, 3)).astype(np.float32) / 255.
            img_input = np.transpose(img_input, (0, 3, 1, 2))

            img_depth = ros_numpy.numpify(depth)  
            
            self.input_data.set_data(DATA_FIELD.TIME_STAMP, np.array([stamp], dtype=np.float64))
            self.input_data.set_data(DATA_FIELD.IMAGE_COLOR, img_color)
            self.input_data.set_data(DATA_FIELD.IMAGE_DEPTH, img_depth)
            self.input_data.set_data(DATA_FIELD.IMAGE_INPUT, img_input)  

            for i in self.obj_idxs:
                self.detection_data[i].set_data(DATA_FIELD.MODEL_FLAG, np.array([True], dtype=np.bool_))
                self.detection_data[i].set_data(DATA_FIELD.IDX, np.array([-1], dtype=np.int32))  

            self.input_data.set_data(DATA_FIELD.PUBLISH_FLAG, np.array([True], dtype=np.bool_))            
                    
    def spin(self):
                
        #rospy.Rate(30)
        rospy.spin()            

def main():

    print(args)

    torch.multiprocessing.set_start_method('spawn')
    prima6d_ros = PRIMA6D_ROS()
    prima6d_ros.spin()
    
if __name__ == "__main__":
    main()
