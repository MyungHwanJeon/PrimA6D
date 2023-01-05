import argparse
import sys
import os
import time
import numpy as np

import torch
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import utils as trt_utils

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model import utils, PrimA6DPPNet, SegmentationNet
from misc.misc import *

parser = argparse.ArgumentParser(
                                description='convert_to_trt')
parser.add_argument('-d', '--dtype', required=False,
                    default="trt16",
                    help="trt16, trt32")   
parser.add_argument('-o', '--obj', required=True,
                    default="5",
                    help="object ID")                                                                    
                                                                 
args = parser.parse_args()
print(args)

os.makedirs('./converted_weight/PrimA6D++/' + args.dtype + '/onnx', exist_ok=True)
os.makedirs('./converted_weight/Segmentation/' + args.dtype + '/onnx', exist_ok=True)
os.makedirs('./converted_weight/PrimA6D++/' + args.dtype + '/engine', exist_ok=True)
os.makedirs('./converted_weight/Segmentation/' + args.dtype + '/engine', exist_ok=True)

def main():
    
    dtype = torch.float16
    if args.dtype == "trt32":
        dtype = torch.float32

    ## segmentation model ##
    model_S = SegmentationNet.SegmentationNet()
    load_weight_all(model_S, pre_trained_path="../Segmentation/trained_weight/obj_" + args.obj + "_S.pth") 
    model_S.to(dtype).cuda()
    model_S.eval()
      
    input_tensor = torch.randn(1, 3, 480, 640, dtype=dtype).cuda()
    trt_utils.convert_to_onnx(model_S, input_tensor, "./converted_weight/Segmentation/" + args.dtype + "/onnx/model_" + args.obj + "_S.onnx")        
    trt_utils.build_engine("./converted_weight/Segmentation/" + args.dtype + "/onnx/model_" + args.obj + "_S.onnx", "./converted_weight/Segmentation/" + args.dtype + "/engine/model_" + args.obj + "_S.engine", dtype=args.dtype)        
            
    ## PrimA6D Model ##
    model_P = PrimA6DPPNet.PrimA6DNet()  
    load_weight_all(model_P, pre_trained_path="../PrimA6D++/trained_weight/obj_" + args.obj + "_P.pth")       
    model_P.to(dtype).cuda() 
    model_P.eval()                                   
    
    input_tensor = torch.randn(1, 3, 64, 64, dtype=dtype).cuda()
    trt_utils.convert_to_onnx(model_P, input_tensor, "./converted_weight/PrimA6D++/" + args.dtype + "/onnx/model_" + args.obj + "_P.onnx")        
    trt_utils.build_engine("./converted_weight/PrimA6D++/" + args.dtype + "/onnx/model_" + args.obj + "_P.onnx", "./converted_weight/PrimA6D++/" + args.dtype + "/engine/model_" + args.obj + "_P.engine", dtype=args.dtype)         
                        
if __name__ == '__main__':
    main()
