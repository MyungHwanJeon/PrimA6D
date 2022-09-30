#off-screen rendering using egl(opengl)
import os
import pyglet
os.environ["PYOPENGL_PLATFORM"] = "egl"
pyglet.options['shadow_window'] = False

import numpy as np
import time
import os
import random
import sys

import cv2

from scipy.spatial import distance
from random import shuffle

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from imgaug.augmenters import Sequential,SomeOf,OneOf,Sometimes,WithColorspace,WithChannels, \
        Noop,Lambda,AssertLambda,AssertShape,Scale,CropAndPad, \
        Pad,Crop,Fliplr,Flipud,Superpixels,ChangeColorspace, PerspectiveTransform, \
        Grayscale,GaussianBlur,AverageBlur,MedianBlur,Convolve, \
        Sharpen,Emboss,EdgeDetect,DirectedEdgeDetect,Add,AddElementwise, \
        AdditiveGaussianNoise,Multiply,MultiplyElementwise,Dropout, \
        CoarseDropout,Invert,Affine,PiecewiseAffine, \
        ElasticTransformation, Fog, Clouds, color, MultiplyHueAndSaturation, MultiplyHue, UniformColorQuantization, AverageBlur, MedianBlur, MotionBlur, \
        AdditiveGaussianNoise, AdditivePoissonNoise, LinearContrast, AdditiveLaplaceNoise, AdditivePoissonNoise, Salt, Pepper

def calc_pts_diameter(pts):

    dists = distance.cdist(pts, pts, 'euclidean')
    diameter = np.max(dists)

    return diameter   

def calc_2d_bbox(xs, ys, im_size):

    bbTL = (max(xs.min() - 1, 0),
            max(ys.min() - 1, 0))
    bbBR = (min(xs.max() + 1, im_size[0] - 1),
            min(ys.max() + 1, im_size[1] - 1))

    return [bbTL[0], bbTL[1], bbBR[0] - bbTL[0], bbBR[1] - bbTL[1]]      

def crop_image_using_bb(img, bb_xywh):

    x, y, w, h = np.array(bb_xywh).astype(np.int32)
    size = int(np.maximum(h, w))
    left = np.maximum(x+w/2-size/2, 0)
    right = np.minimum(x+w/2+size/2, img.shape[1])
    top = np.maximum(y+h/2-size/2, 0)
    bottom = np.minimum(y+h/2+size/2, img.shape[0])

    cropped_img = img[int(top):int(bottom), int(left):int(right)]

    return cropped_img, np.array([int(left), int(top), int(right)-int(left), int(bottom)-int(top)])

def make_padding_bb(img, bb_xywh, render_dimension=[640, 480, 3], pad_factor=1.3):
    
    x, y, w, h = np.array(bb_xywh).astype(np.int32)                               
    size = int(np.maximum(h, w) * pad_factor)                

    if x+w/2-size/2 < 0:
        left = 0
        right = np.minimum(x+w/2+size/2, render_dimension[0])
    elif x+w/2+size/2 > render_dimension[0]:
        right = render_dimension[0]
        left = np.maximum(x+w/2-size/2, 0)
    else:
        left = np.maximum(x+w/2-size/2, 0)
        right = np.minimum(x+w/2+size/2, render_dimension[0])
        
    if y+h/2-size/2 < 0:
        top = 0
        bottom = np.minimum(y+h/2+size/2, render_dimension[1])
    elif y+h/2+size/2 > render_dimension[1]:
        bottom = render_dimension[1]
        top = np.maximum(y+h/2-size/2, 0)
    else:
        top = np.maximum(y+h/2-size/2, 0)
        bottom = np.minimum(y+h/2+size/2, render_dimension[1])            

    return np.array([int(left), int(top), int(right)-int(left), int(bottom)-int(top)]) 

def augment_squares(mask, max_occl=0.4):

    size_percent = random.uniform(0.01, 0.5)        
    aug = Sequential([Sometimes(0.3, CoarseDropout(p=max_occl, size_percent=size_percent) )])

    inverted_mask = np.invert(mask)
    number_of_obj_pixels = np.count_nonzero(inverted_mask==1,axis=(0,1))

    dropout_masks = aug.augment_images(inverted_mask)
    dropout_number_of_obj_pixels = np.count_nonzero(dropout_masks==1,axis=(0,1))             

    while number_of_obj_pixels*(1-max_occl) < dropout_number_of_obj_pixels:
        dropout_masks = aug.augment_images(inverted_mask)
        dropout_number_of_obj_pixels = np.count_nonzero(dropout_masks==1,axis=(0,1))
    
    return np.invert(dropout_masks)


if __name__ == '__main__':
    main()        
