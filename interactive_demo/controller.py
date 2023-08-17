# Adapted from: https://github.com/SamsungLabs/ritm_interactive_segmentation/blob/master/interactive_demo/controller.py
import torch
import numpy as np
from tkinter import messagebox
import cv2
from detectron2.data import transforms as T
import copy
from dynamite.utils.misc import color_map

class InteractiveController:
    def __init__(self, model, update_image_callback, cfg):

        self.transforms = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
       
        self._result_masks = None
        self._inputs = {}
        self.image = None
        self.predictor = model
        self.update_image_callback = update_image_callback

        self.fg_orig_coords = []
        self.bg_orig_coords = []

    def set_image(self, image):
        self.image = copy.deepcopy(image)
       
        h,w = self.image.shape[:2]
        self._inputs["height"] = h
        self._inputs["width"] = w
        img=  self.transforms.get_transform(self.image).apply_image(self.image)
        self._inputs["image"] = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))

        self.new_h, self.new_w = self._inputs["image"].shape[-2:]
        self.rh = self.new_h/h
        self.rw = self.new_w/w

         # store input for transformer decoder
        self.device = None
        self.click_counts = 0
        self.processed_results = None
        self.images=None
        self.num_insts = None
        self.features = None
        self.mask_features = None
        self.multi_scale_features=None
        self.num_clicks_per_object = None
        self.fg_coords = None
        self.bg_coords = None
        self.max_timestamp = [1]
        
        self.fg_orig_coords = []
        self.bg_orig_coords = []
        self._result_masks = None
        self.object_count = 0
        self.update_image_callback(reset_canvas=True)

    def add_click(self, x, y, bg_click =False, inst_num=1):

        if inst_num==0:
            print("add an instance")
            self.update_image_callback()
            return 
        
        new_y = y*self.rh
        new_x = x*self.rw
        if bg_click:
            
            if self.bg_coords[0]:
                self.bg_coords[0].extend([[new_y, new_x,self.click_counts]])
            else:
                self.bg_coords[0] = [[new_y, new_x,self.click_counts]]
            
            self.bg_orig_coords.append([x, y, self.click_counts])
        else:
            if not self.fg_coords:
                self.fg_orig_coords = [[[x, y, self.click_counts]]]
               
                self._inputs["fg_click_coords"] = [[[new_y, new_x, self.click_counts]]]
                self._inputs["bg_click_coords"] = None
              
                self._inputs["num_clicks_per_object"] = [1]
            elif inst_num <= len(self.num_clicks_per_object[0]):
                self.num_clicks_per_object[0][inst_num-1] += 1
                self.fg_coords[0][inst_num-1].extend([[new_y, new_x,self.click_counts]])
                self.fg_orig_coords[inst_num-1].extend([[x, y, self.click_counts]])
            else:
                self.num_clicks_per_object[0].append(1)
                self.fg_coords[0].append([[new_y, new_x,self.click_counts]])
                self.fg_orig_coords.append([[x, y, self.click_counts]])
                
        self.click_counts+=1
        if self.num_insts is not None:
            self.num_insts[0] = len(self.num_clicks_per_object[0])
      
        inputs = [self._inputs]

        if self.features is None:
            (processed_results, outputs, self.images,
            self.num_insts, self.features, self.mask_features,
            self.multi_scale_features,
            self.num_clicks_per_object, self.fg_coords,
            self.bg_coords) = self.predictor(inputs, max_timestamp=self.max_timestamp)
            self.device = self.images.tensor.device
        else:
            (processed_results, outputs, self.images,
            self.num_insts, self.features, self.mask_features,
            self.multi_scale_features,
            self.num_clicks_per_object, self.fg_coords,
            self.bg_coords) = self.predictor(inputs, self.images, self.num_insts,
                                                self.features, self.mask_features,
                                                self.multi_scale_features, 
                                                self.num_clicks_per_object,
                                                self.fg_coords, self.bg_coords,
                                                max_timestamp=self.max_timestamp)

        self._result_masks = processed_results[0]['instances'].pred_masks

        # We don't overwrite current mask until commit
        self.last_masks = self._result_masks

        self.update_image_callback()
    
    def undo_click(self):
        pass

    def reset_clicks(self):
        self.set_image(self.image)
        self.update_image_callback(reset_clicks=True)

    @property
    def result_masks(self):
        if self._result_masks is not None:
            result_masks = self._result_masks.clone()
        else:
            result_masks = self._result_masks
        return result_masks

    
    def get_visualization(self, alpha_blend=0.3, click_radius=3, reset_clicks=False,show_only_masks=False):
       
        if self.image is None:
            return None, None

        result_masks_for_vis = self.result_masks
        image = np.asarray(copy.deepcopy(self.image))
        if (result_masks_for_vis is None) or (reset_clicks):
            return image, None

        result_masks_for_vis = result_masks_for_vis.to(device ='cpu')
        
        pred_masks =np.asarray(result_masks_for_vis,dtype=np.uint8)
        c = []
        for i in range(pred_masks.shape[0]):
            c.append(color_map[i]/255.0)
       
        for i in range(pred_masks.shape[0]):
            image = self.apply_mask(image, pred_masks[i], c[i],alpha_blend)
        total_colors = len(color_map)-1
        
        point_clicks_map = np.ones_like(image)*255
        if not show_only_masks:
            if len(self.fg_orig_coords):
                for j, fg_coords_per_mask in enumerate(self.fg_orig_coords):
                    for i, coords in enumerate(fg_coords_per_mask):
                        color = np.array(color_map[total_colors-5*j-4], dtype=np.uint8)
                        color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
                        image = cv2.circle(image, (int(coords[0]), int(coords[1])), click_radius, tuple(color), -1)
            
            if len(self.bg_orig_coords):
                for i, coords in enumerate(self.bg_orig_coords):
                    color = np.array([255,0,0], dtype=np.uint8)
                    color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
                    image = cv2.circle(image, (int(coords[0]), int(coords[1])), click_radius, tuple(color), -1)
        return image, point_clicks_map

    def apply_mask(self, image, mask, color, alpha=0.5):
        for c in range(3):
            image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
        return image


def get_color_from_map(index):
    color_c='#%02x%02x%02x' % (color_map[index][0], color_map[index][1], color_map[index][2])
    return color_c
    