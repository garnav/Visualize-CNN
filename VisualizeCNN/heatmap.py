# heatmap.py
# Arnav Ghosh
# 27th March 2020

import copy
import cv2
import numpy as np
import plotly.graph_objects as go
from scipy import ndimage

class HeatMap(object):

    # image : (H, W, C)
    # heatmap : (H, W) with pixels in [0.0, 1.0]
    def __init__(self, image, heatmap):
        self.image = image
        self.heatmap = heatmap.squeeze()

    def heatmap_spots(self, threshold):
        # threshold to create features
        ret, heatmap = cv2.threshold(self.heatmap.copy(), threshold, 1, cv2.THRESH_BINARY)
        
        # dilate to connect closeby components, widen scope of object
        heatmap = cv2.dilate(heatmap, np.ones((5, 5)), iterations=1) 
        
        # label heatmap pixels as being part of the same 'object'
        structure = np.ones((3, 3)) # diagonal connections included
        heatmap_labels, num_feat = ndimage.measurements.label(heatmap, structure=structure)
        return heatmap_labels, num_feat

    # (cent_x, cent_y), (minor_ax, major_ax), angle
    def get_ellipse(self, th_image):
        _, contours, _ = cv2.findContours(th_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # retaining max area contour
        contour_areas = np.array([cv2.contourArea(contour) for contour in contours])
        ellipse = cv2.fitEllipse(contours[np.argmax(contour_areas)])
        return ellipse

    # img_dim : (h, w)
    def draw_ellipse(self, ellipses):
        image = np.zeros(self.image.shape)
        for (x, y), (min_ax, maj_ax), angle in ellipses:
            image = cv2.ellipse(image, (int(x), int(y)), (int(min_ax/2), int(maj_ax/2)), int(angle), 
                                0, 360, 1, -1)
        return image 

    # minimum will be a factor of the upscaling we do
    def heatmap_ellipses(self, threshold):
        heatmap_labels, num_feat = self.heatmap_spots(threshold)

        all_ellipses = []
        for i in range(1, num_feat + 1):
            th_image = heatmap_labels.copy()
            th_image[th_image != i] = 0
            th_image[th_image == i] = 1
            ellipse = self.get_ellipse(th_image.astype(np.uint8))

            all_ellipses.append(ellipse)

        return all_ellipses

    # TODO: Use of IOU of Ellipses or of Areas?
    # Heatmaps Should come in as the same size
    def compare_heatmaps(self, hmap2, threshold):
        hmap1_ellipse = self.heatmap_ellipses(threshold)
        hmap2_ellipse = hmap2.heatmap_ellipses(threshold)

        if (len(hmap1_ellipse) == 0) or (len(hmap2_ellipse) == 0):
            return 0

        draw_ellipses1 = self.draw_ellipse(hmap1_ellipse)
        draw_ellipses2 = self.draw_ellipse(hmap2_ellipse)
        covered_region1 = draw_ellipses1 != 0
        covered_region2 = draw_ellipses2 != 0

        intersection = np.sum(np.logical_and(covered_region1, covered_region2))
        union = np.sum(np.logical_or(covered_region1, covered_region2))
        return intersection / union

    # TODO: Check Literature on segmentation & use principles from there
    # Multiple focusses? Got Max
    # Abnormal ratio of focusses?
    # Automatically adjusting window size based on heatmap? D
    # https://nrsyed.com/tag/object-detection/
    def get_focus_window(self, threshold):
        heatmap_labels, num_feat = self.heatmap_spots(threshold)

        # get largest region (TODO: Get most compact instead?)
        label_idx = np.argmax([np.sum(heatmap_labels == i) for i in range(1, num_feat + 1)]) + 1
        y_locs, x_locs = np.where(heatmap_labels == label_idx)
        x_left, y_top = np.min(x_locs), np.min(y_locs)
        x_right, y_bot = np.max(x_locs), np.max(y_locs)

        region = self.image[y_top : y_bot + 1, x_left : x_right + 1,:] # TODO: Blur Image or weight by og hmap
        return region

    # TODO: What if you end up with a scenario where it's superfocussed so everything but that region is black? [set minimum on hmap]
    def get_focus_image(self, threshold):
        mod_heatmap = self.heatmap[:, :]
        mod_heatmap[mod_heatmap < threshold] = threshold
        return self.image * np.expand_dims(mod_heatmap, axis=2)

    def plot_heatmap(self):
        fig = go.Figure()
        fig.add_trace(go.Image(z=self.image))
        fig.add_trace(go.Heatmap(z=self.heatmap, 
                                 type = 'heatmap',
                                 colorscale = 'jet', 
                                 opacity=0.7))
        return fig
