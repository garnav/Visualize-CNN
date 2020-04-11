import copy
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class OcclusionCam(object):
    
    def __init__(self, model, occ_height, occ_width, color):
        self.model = model
        self.model.eval()

        self.occ_height = occ_height
        self.occ_width = occ_width
        self.set_color(color)

    def set_occ_dim(self, occ_height, occ_width):
        self.occ_height = occ_height
        self.occ_width = occ_width

    def set_color(self, color):
        self.color = torch.Tensor(color)

    def occlude_image(self, image):
        img_height, img_width, _ = image.shape
        y_occ, x_occ = int(img_height / self.occ_height), int(img_width / self.occ_width)
        num_occ = y_occ * x_occ
        top_y, top_x = int((img_height % self.occ_height)/2), int((img_width % self.occ_width)/2)
        
        all_images = image.repeat(num_occ,1,1,1)
        occ_locs = np.zeros((num_occ, 4), dtype=np.int32)

        for r in range(y_occ):
            for c in range(x_occ):
                i = int((r * x_occ) + c)
                start_y, start_x = top_y + (r * self.occ_height), top_x + (c * self.occ_width)
                occ_locs[i, :] = [start_x, start_y, start_x + self.occ_width, start_y + self.occ_height]
                
                if r == 0:
                    occ_locs[i, 1] = 0
                elif r == (y_occ - 1):
                    occ_locs[i, 3] = img_height
                if c == 0:
                    occ_locs[i, 0] = 0
                elif c == (x_occ - 1):
                    occ_locs[i, 2] = img_width
                
                all_images[i, occ_locs[i, 1] : occ_locs[i, 3], occ_locs[i, 0]: occ_locs[i, 2], :] = self.color
                
        return all_images, occ_locs

    # image : (H, W, C)
    def __call__(self, image):
        occ_images, occ_locs = self.occlude_image(image)
        occ_images = occ_images.permute(0, 3, 1, 2)
        occ_dataset = TensorDataset(occ_images.double(), 
                                    torch.Tensor(occ_locs).int())
        occ_loader = DataLoader(occ_dataset, batch_size=4, shuffle=False)

        heatmap = None
        with torch.set_grad_enabled(False):
            for _, (inputs, locs) in enumerate(occ_loader):
                outputs = self.model(inputs)

                if heatmap is None:
                    heatmap = np.zeros((image.shape[0], image.shape[1], outputs.size()[1]))

                for i in range(locs.shape[0]):
                    heatmap[locs[i, 1] : locs[i, 3], locs[i, 0] : locs[i, 2], :] = outputs[i, :].cpu().data.numpy()

        return heatmap

# TODO:
# Currently uses stride as height, width (assuems that the patch is big enough to cover small features, small enough to not cause too much change)
# ensure images are transformed before passing