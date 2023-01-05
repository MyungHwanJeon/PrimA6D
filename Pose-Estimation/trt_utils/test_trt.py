import time
import numpy as np
import cupy as cp

import torch
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import utils

import cv2

def main():
    
    t1 = time.time()
    
    ctx = cuda.Device(0).make_context()
    cp_dev = cp.cuda.Device(0)
    cp_dev.use()
    
    ## test segmentation model ##
    s_engine = utils.load_engine("./converted_weight/Segmentation/trt16/engine/model_5_S.engine")            
    s_inputs, s_outputs, s_bindings, s_stream = utils.allocate_buffers(s_engine)
    s_context = s_engine.create_execution_context()
    
    s_img = cv2.imread("./test_s.png", cv2.IMREAD_COLOR)  
    s_img = s_img.reshape((1, 480, 640, 3)).astype(np.float32) / 255.
    s_img = np.transpose(s_img, (0, 3, 1, 2))
    
    for i in range(10):
        t1 = time.time()
        np.copyto(s_inputs[0].host, s_img.ravel())  
        s_output = utils.do_inference_v2(s_context, bindings=s_bindings, inputs=s_inputs, outputs=s_outputs, stream=s_stream)
        print(i, time.time() - t1)

    s_pred = s_output[0].reshape((1, 2, 480, 640))
    s_pred_gpu = cp.array(s_pred)
    s_pred_gpu = cp.argmax(s_pred_gpu, axis=1)
    s_pred = s_pred_gpu.get()
    
    cv2.imshow("s", s_pred.astype(np.uint8).reshape(480, 640, 1)*255)
    cv2.waitKey(0)
        
    ## test PrimA6D model ##    
    p_engine = utils.load_engine("./converted_weight/PrimA6D++/trt16/engine/model_5_P.engine")            
    p_inputs, p_outputs, p_bindings, p_stream = utils.allocate_buffers(p_engine)
    p_context = p_engine.create_execution_context()
    
    p_img = cv2.imread("./test_p.png", cv2.IMREAD_COLOR)  
    p_img = p_img.reshape((1, 64, 64, 3)).astype(np.float32) / 255.
    p_img = np.transpose(p_img, (0, 3, 1, 2))
    
    for i in range(10):
        t1 = time.time()
        np.copyto(p_inputs[0].host, p_img.ravel())  
        p_output = utils.do_inference_v2(p_context, bindings=p_bindings, inputs=p_inputs, outputs=p_outputs, stream=p_stream)
        print(i, time.time() - t1)

    primitive_out = np.transpose(p_output[0].reshape((9, 64, 64)), (1, 2, 0))
    keypoint_out = p_output[1].reshape((42, 2))
    uncertainty_out = p_output[2].reshape((3))
    
    cv2.imshow("p_x", primitive_out[:, :, :3].astype(np.float32))
    cv2.imshow("p_y", primitive_out[:, :, 3:6].astype(np.float32))
    cv2.imshow("p_z", primitive_out[:, :, 6:9].astype(np.float32))
    cv2.waitKey(0)
               
    ctx.pop()    



if __name__ == '__main__':
    main()

