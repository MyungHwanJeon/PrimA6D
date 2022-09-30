import os
import pyglet
os.environ["PYOPENGL_PLATFORM"] = "egl"
pyglet.options['shadow_window'] = False

import numpy as np
import time
import glob
import random
import sys
from random import shuffle

import cv2

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import transform
from dataset_utils import *
import config

import imgaug
import imgaug.augmenters as iaa
 
import torch 

from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    Textures,
    TexturesUV,
    TexturesVertex,
    PerspectiveCameras,
    MeshRendererWithFragments,
    TexturesAtlas,
    TexturesUV
)
from pytorch3d.ops.perspective_n_points import efficient_pnp


class synthetic_dataset(object):
    
    def __init__(self, dataset="LINEMOD", obj_idx=1):
    
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
             
        self.bkg_files = [os.path.join(config.background_images_path,  x) for x in os.listdir(config.background_images_path)] 

        self.dataset = dataset
        self.obj_idx = obj_idx
        
        #the path of current file
        current_path = os.path.dirname(os.path.realpath(__file__))
        
        if self.dataset == "LINEMOD":            
            self.K = np.array([ [572.4114, 0.0,       325.2611], 
                                [0.0,      573.57043, 242.04899], 
                                [0.0,      0.0,       1.0]])
            self.obj_model_path = current_path + '/3d_model/LINEMOD/obj_' + str(self.obj_idx) + '.ply'
        elif self.dataset == "YCB":            
            self.K = np.array([ [1066.778,  0.0,       312.9869], 
                                [0.0,       1067.487,  241.3109], 
                                [0.0,       0.0,        1.0]])
            self.obj_model_path = current_path + '/3d_model/YCB/obj_' + str(self.obj_idx) + '/textured_reduced.obj'            
        

        ## object model ##
        verts, faces, aux = load_obj(self.obj_model_path)
        verts = verts * 1000.
        self.obj_diameter = calc_pts_diameter(verts.data.cpu().numpy().astype(np.float32))
        print("obj diameter :", self.obj_diameter)
        
        if self.obj_idx is 5 and self.dataset is "YCB":
            verts = verts.permute(1, 0)
            verts = torch.mm(torch.from_numpy(transform.euler_matrix(0, 0, 23*(np.pi/180.))[:3, :3].astype(np.float32)), verts)
            verts = verts.permute(1, 0)
                
        tex_maps = aux.texture_images
        if tex_maps is not None and len(tex_maps) > 0:
            verts_uvs = aux.verts_uvs.to(self.device) 
            faces_uvs = faces.textures_idx.to(self.device)
            image = list(tex_maps.values())[0].to(self.device)[None]
            tex = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image)
        self.obj_mesh = Meshes(verts=[verts.to(self.device)], faces=[faces.verts_idx.to(self.device)], textures=tex)
        
        ## primitive 3-axis model ##
        self.primitive_model_path = current_path + '/3d_model/PRIMITIVE/3axis.obj'         
        verts, faces, aux = load_obj(self.primitive_model_path)
        self.primitive_diameter = calc_pts_diameter(verts.data.cpu().numpy().astype(np.float32))
        self.primitive_scale = self.obj_diameter*0.5 / self.primitive_diameter
        verts *= self.primitive_scale
        print("3 axis diameter :", self.primitive_scale)
        tex_maps = aux.texture_images
        if tex_maps is not None and len(tex_maps) > 0:
            verts_uvs = aux.verts_uvs.to(self.device) 
            faces_uvs = faces.textures_idx.to(self.device)
            image = list(tex_maps.values())[0].to(self.device)[None]
            tex = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image)
        self.primitive_mesh = Meshes(verts=[verts.to(self.device)], faces=[faces.verts_idx.to(self.device)], textures=tex)   
                                
        ## primitive 1-axis-x model ##
        self.primitive_1axis_x_model_path = current_path + '/3d_model/PRIMITIVE/1axis_x.obj'         
        verts, faces, aux = load_obj(self.primitive_1axis_x_model_path)
        self.primitive_diameter = calc_pts_diameter(verts.data.cpu().numpy().astype(np.float32))
        self.primitive_scale = self.obj_diameter*0.5 / self.primitive_diameter
        verts *= self.primitive_scale
        print("1 axis x diameter :", self.primitive_scale)
        tex_maps = aux.texture_images
        if tex_maps is not None and len(tex_maps) > 0:
            verts_uvs = aux.verts_uvs.to(self.device) 
            faces_uvs = faces.textures_idx.to(self.device)
            image = list(tex_maps.values())[0].to(self.device)[None]
            tex = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image)
        self.primitive_1axis_x_mesh = Meshes(verts=[verts.to(self.device)], faces=[faces.verts_idx.to(self.device)], textures=tex)           
        
        ## primitive 1-axis-y model ##
        self.primitive_1axis_y_model_path = current_path + '/3d_model/PRIMITIVE/1axis_y.obj'         
        verts, faces, aux = load_obj(self.primitive_1axis_y_model_path)
        self.primitive_diameter = calc_pts_diameter(verts.data.cpu().numpy().astype(np.float32))
        self.primitive_scale = self.obj_diameter*0.5 / self.primitive_diameter
        verts *= self.primitive_scale
        print("1 axis y diameter :", self.primitive_scale)
        tex_maps = aux.texture_images
        if tex_maps is not None and len(tex_maps) > 0:
            verts_uvs = aux.verts_uvs.to(self.device) 
            faces_uvs = faces.textures_idx.to(self.device)
            image = list(tex_maps.values())[0].to(self.device)[None]
            tex = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image)
        self.primitive_1axis_y_mesh = Meshes(verts=[verts.to(self.device)], faces=[faces.verts_idx.to(self.device)], textures=tex)    
        
        ## primitive 1-axis-z model ##
        self.primitive_1axis_z_model_path = current_path + '/3d_model/PRIMITIVE/1axis_z.obj'         
        verts, faces, aux = load_obj(self.primitive_1axis_z_model_path)
        self.primitive_diameter = calc_pts_diameter(verts.data.cpu().numpy().astype(np.float32))
        self.primitive_scale = self.obj_diameter*0.5 / self.primitive_diameter
        verts *= self.primitive_scale
        print("1 axis z diameter :", self.primitive_scale)
        tex_maps = aux.texture_images
        if tex_maps is not None and len(tex_maps) > 0:
            verts_uvs = aux.verts_uvs.to(self.device) 
            faces_uvs = faces.textures_idx.to(self.device)
            image = list(tex_maps.values())[0].to(self.device)[None]
            tex = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image)
        self.primitive_1axis_z_mesh = Meshes(verts=[verts.to(self.device)], faces=[faces.verts_idx.to(self.device)], textures=tex)                         
    
        ## renderer setting ##
        raster_settings = RasterizationSettings(image_size=[640, 640], blur_radius=0.000, faces_per_pixel=1,)                           
        self.renderer = MeshRendererWithFragments(
                                                rasterizer=MeshRasterizer(raster_settings=raster_settings),
                                                shader=SoftPhongShader(device=self.device)
                                                )
                                                
    def render_synthetic(self, rot, tra, K=None):
              
        ## set 3D object pose ##
        camera_offset = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32).to(self.device)
        camera_offset = camera_offset.view(1, 3, 3)
        rot = torch.from_numpy(rot.astype(np.float32)).view(1, 3, 3).to(self.device)                        
        rot = torch.bmm(camera_offset, rot)
        rot = torch.inverse(rot)
        
        camera_offset = torch.tensor([-1, -1, 1], dtype=torch.float32).to(self.device)
        camera_offset = camera_offset.view(1, 3)
        tra = torch.from_numpy(tra.astype(np.float32)).view(1, 3).to(self.device)
        tra = camera_offset * tra
        
        
        if K is not None:
            self.K = K.reshape(3, 3)
        
        cameras = PerspectiveCameras(
                                    focal_length=np.array([self.K[0, 0], self.K[1, 1]]).reshape((1, 2)), 
                                    principal_point=np.array([self.K[0, 2], self.K[1, 2]]).reshape((1, 2)), 
                                    image_size=np.array([640, 640]).reshape((1, 2)), 
                                    device=self.device, 
                                    R=rot, 
                                    T=tra,
                                    )
        cameras.zfar = 10000
        cameras.znear = 1
           
        ## render image

        ambient_rand = (random.random() - 0.5) * 0.6 + 0.5
        diffuse_rand = (random.random() - 0.5) * 0.4 + 0.4
        specular_rand = (random.random() - 0.5) * 0.4 + 0.3
        x_rand = tra[0, 0].data.cpu().numpy() + np.random.standard_normal(1)*self.obj_diameter*2
        y_rand = tra[0, 1].data.cpu().numpy() + np.random.standard_normal(1)*self.obj_diameter*2
        z_rand = tra[0, 2].data.cpu().numpy() + np.random.standard_normal(1)*self.obj_diameter*2
        lights = PointLights(
                            ambient_color=[[ambient_rand, ambient_rand, ambient_rand]], 
                            diffuse_color=[[diffuse_rand, diffuse_rand, diffuse_rand]], 
                            specular_color=[[specular_rand, specular_rand, specular_rand]], 
                            location=[[x_rand[0], y_rand[0], z_rand[0]]],
                            device=self.device
                            )
        obj_color, obj_depth = self.renderer(self.obj_mesh, cameras=cameras, lights=lights)        
        obj_color = obj_color[0, 0:480, 0:640, 0:3].view(480, 640, 3)*255.
        obj_depth = obj_depth.zbuf[0, 0:480, 0:640, 0].view(480, 640)
        obj_color = obj_color.data.cpu().numpy().astype(np.uint8)
        obj_color = cv2.cvtColor(obj_color, cv2.COLOR_BGR2RGB) 
        obj_depth = obj_depth.data.cpu().numpy()
        obj_mask = obj_depth != -1
        
        lights = PointLights(ambient_color=[[0.8, 0.8, 0.8]], diffuse_color=[[0.8, 0.8, 0.8]], specular_color=[[0.8, 0.8, 0.8]], location=[[0, 0, 0]], device=self.device)
        gt_color, gt_depth = self.renderer(self.obj_mesh, cameras=cameras, lights=lights)
        gt_color = gt_color[0,0:480, 0:640, 0:3].view(480, 640, 3)*255.
        gt_depth = gt_depth.zbuf[0, 0:480, 0:640, 0].view(480, 640)
        gt_color = gt_color.data.cpu().numpy().astype(np.uint8)
        gt_color = cv2.cvtColor(gt_color, cv2.COLOR_BGR2RGB) 
        gt_depth = gt_depth.data.cpu().numpy()
        
        lights = PointLights(ambient_color=[[1.0, 1.0, 1.0]], diffuse_color=[[1.0, 1.0, 1.0]], specular_color=[[1.0, 1.0, 1.0]], location=[[0, 0, 0]], device=self.device)
        primitive_color, _ = self.renderer(self.primitive_mesh, cameras=cameras, lights=lights)
        primitive_color = primitive_color[0, 0:480, 0:640, 0:3].view(480, 640, 3)*255        
        primitive_color = primitive_color.data.cpu().numpy().astype(np.uint8)  
        primitive_color = cv2.cvtColor(primitive_color, cv2.COLOR_BGR2RGB) 
        
        lights = PointLights(ambient_color=[[1.0, 1.0, 1.0]], diffuse_color=[[1.0, 1.0, 1.0]], specular_color=[[1.0, 1.0, 1.0]], location=[[0, 0, 0]], device=self.device)
        primitive_x_color, _ = self.renderer(self.primitive_1axis_x_mesh, cameras=cameras, lights=lights)
        primitive_x_color = primitive_x_color[0, 0:480, 0:640, 0:3].view(480, 640, 3)*255        
        primitive_x_color = primitive_x_color.data.cpu().numpy().astype(np.uint8)  
        primitive_x_color = cv2.cvtColor(primitive_x_color, cv2.COLOR_BGR2RGB) 
        
        lights = PointLights(ambient_color=[[1.0, 1.0, 1.0]], diffuse_color=[[1.0, 1.0, 1.0]], specular_color=[[1.0, 1.0, 1.0]], location=[[0, 0, 0]], device=self.device)
        primitive_y_color, _ = self.renderer(self.primitive_1axis_y_mesh, cameras=cameras, lights=lights)
        primitive_y_color = primitive_y_color[0, 0:480, 0:640, 0:3].view(480, 640, 3)*255        
        primitive_y_color = primitive_y_color.data.cpu().numpy().astype(np.uint8)  
        primitive_y_color = cv2.cvtColor(primitive_y_color, cv2.COLOR_BGR2RGB) 
        
        lights = PointLights(ambient_color=[[1.0, 1.0, 1.0]], diffuse_color=[[1.0, 1.0, 1.0]], specular_color=[[1.0, 1.0, 1.0]], location=[[0, 0, 0]], device=self.device)
        primitive_z_color, _ = self.renderer(self.primitive_1axis_z_mesh, cameras=cameras, lights=lights)
        primitive_z_color = primitive_z_color[0, 0:480, 0:640, 0:3].view(480, 640, 3)*255        
        primitive_z_color = primitive_z_color.data.cpu().numpy().astype(np.uint8)  
        primitive_z_color = cv2.cvtColor(primitive_z_color, cv2.COLOR_BGR2RGB)    

       
        obj_ys, obj_xs = np.nonzero(obj_mask > 0)
        if obj_xs.shape[0] == 0 or obj_ys.shape[0] == 0:
            return False, False, False, False, False, False, False, False, False
                                
        #shuffle(self.bkg_files)    
        rand_bkg_file = self.bkg_files[np.random.choice(len(self.bkg_files), 1, replace=False)[0]]
        rand_bkg = cv2.imread(rand_bkg_file)
        rand_bkg = cv2.resize(rand_bkg, (obj_color.shape[1], obj_color.shape[0]), interpolation=cv2.INTER_LINEAR)          

        obj_color[np.invert(obj_mask.astype(np.bool))] = rand_bkg[np.invert(obj_mask.astype(np.bool))]
        
        return True, obj_color, obj_depth, obj_mask, gt_color, primitive_color, primitive_x_color, primitive_y_color, primitive_z_color
    
def main():

    dataset = synthetic_dataset(dataset="YCB", obj_idx=5)       
    
    #out_video1 = cv2.VideoWriter('./1.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))
    #out_video2 = cv2.VideoWriter('./2.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))
    #out_video3 = cv2.VideoWriter('./3.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))
    #out_video4 = cv2.VideoWriter('./4.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))
    #out_video5 = cv2.VideoWriter('./5.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))
    
    for i in range(360): 
        print(i)
        
        state, obj_color, obj_depth, obj_mask, gt_color, primitive_color, primitive_1axis_x_color, primitive_1axis_y_color, primitive_1axis_z_color = dataset.render_synthetic(transform.euler_matrix(70*(3.141592/180.), i*(3.141592/180.), 0)[0:3, 0:3], np.array([0,0, 1000]))

        cv2.imshow("obj_color", obj_color)
        cv2.imshow("gt_color", gt_color)
        cv2.imshow("primitive_color", primitive_color)
        cv2.imshow("primitive_1axis_x_color", primitive_1axis_x_color)
        cv2.imshow("primitive_1axis_y_color", primitive_1axis_y_color)
        cv2.imshow("primitive_1axis_z_color", primitive_1axis_z_color)        
        cv2.waitKey(10)
        
        #out_video1.write(gt_color)
        #out_video2.write(primitive_color)
        #out_video3.write(primitive_1axis_x_color)
        #out_video4.write(primitive_1axis_y_color)
        #out_video5.write(primitive_1axis_z_color)        
        
    #out_video1.release()
    #out_video2.release()
    #out_video3.release()
    #out_video4.release()
    #out_video5.release()
    
        

if __name__ == '__main__':
    main()        
