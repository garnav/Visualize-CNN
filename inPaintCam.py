# inPaintCam.py
# Arnav Ghosh
# 22nd March 2020

import copy
import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

class InPaintCam(object):

    def __init__(self, model, kernel_size, paint_flag):
        self.model = copy.deepcopy(model)
        #self.model.eval()
        self.kernel_size = kernel_size
        self.paint_flag = paint_flag

    def set_kernel(self, kernel_size):
        self.kernel_size = kernel_size
    
    def set_flag(self, paint_flag):
        self.paint_flag = paint_flag
    
    # image is type uint8
    # image_mask is H,W,1 with 1s
    def __call__(self, image, image_mask):
        mod_image = cv2.inpaint(image, 
                                cv2.UMat(image_mask.astype(np.uint8)), 
                                self.kernel_size, 
                                self.paint_flag).get()

        with torch.set_grad_enabled(False):
            mod_image = transforms.ToTensor()(mod_image).unsqueeze(0)
            output = self.model(mod_image)

            return output


# img_mask = np.zeros((256,256, 1))
# img_mask[150:175, 100:125, 0] = 1
# dst = cv2.inpaint(int_image, 
#                   cv2.UMat(img_mask.astype(np.uint8)), 
#                   3, 
#                   cv2.INPAINT_NS)
# plt.imshow(dst.get())