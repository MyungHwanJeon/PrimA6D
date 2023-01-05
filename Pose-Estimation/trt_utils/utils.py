import argparse
import os

import numpy as np

import torch

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def GiB(val):
    return val * 1 << 30
    
def convert_to_onnx(model, input_tensor, file_name):

    torch.onnx.export(
                model,  # model being run                        
                input_tensor,    # model input (or a tuple for multiple inputs) 
                file_name, # where to save the model  
                export_params=True,  # store the trained parameter weights inside the model file 
                opset_version=13,    # the ONNX version to export the model to 
                do_constant_folding=True,  # whether to execute constant folding for optimization 
                input_names = ['modelInput'],   # the model's input names 
                output_names = ['modelOutput'], # the model's output names 
                #dynamic_axes={'modelInput' : {0 : 'batch_size'}, 'modelOutput' : {0 : 'batch_size'}}    # variable length axes 
                ) 
                                
def build_engine(onnx_file_path, engine_file_path, dtype="fp16"):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    builder = trt.Builder(TRT_LOGGER)
    
    network_flag = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    network = builder.create_network(network_flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    runtime = trt.Runtime(TRT_LOGGER)

    # Parse model file
    print('Loading ONNX file from path {}...'.format(onnx_file_path))
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print('Completed parsing of ONNX file')

    # Print input info
    #print('Network inputs:')
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        #print(tensor.name, trt.nptype(tensor.dtype), tensor.shape)

    #network.get_input(0).shape = [10, 1]
    #network.get_input(1).shape = [10, 1, 1, 16]
    #network.get_input(2).shape = [6, 1]
    #network.get_input(3).shape = [6, 1, 1, 16]

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.REFIT)    
    config.max_workspace_size = 1 << 28  # 256MiB
    
    if dtype == "fp32":
        config.set_flag(trt.BuilderFlag.TF32)
    elif dtype == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
    plan = builder.build_serialized_network(network, config)
    engine = runtime.deserialize_cuda_engine(plan)
    
    with open(engine_file_path, "wb") as f:
        f.write(plan)
        
    print("Completed creating Engine")        

def load_engine(engine_file_path):

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            return runtime.deserialize_cuda_engine(f.read())      
                        
# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]            
