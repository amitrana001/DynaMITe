import torchvision
import torch
import numpy as np
import cv2
from dynamite.data.dataset_mappers.utils import create_circular_mask
import os
from dynamite.utils.misc import color_map

class Clicker:

    def __init__(self, inputs, sampling_strategy =1, click_radius = 5):
        
        # self.model = model
        self.inputs = inputs
        
        self.click_radius = click_radius
        #TO DO
        # self.normalize_time= normalize_time
        self.max_timestamps = None
        
        # For sampling next click
        self.sampling_strategy = sampling_strategy
        self.not_clicked_map = None
       
        self.click_counts = 0
        self.click_sequence = None

        self.num_insts = []
        self.num_clicks_per_object = []
        self.fg_coords = []
        self.bg_coords = []
        self.fg_orig_coords = []
        self.bg_orig_coords = []
        self.pred_masks = None
        self._set_gt_info()
        
        self.max_timestamps = [self.num_instances-1]
    
    def _set_gt_info(self):

        self.gt_masks = self.inputs[0]['instances'].gt_masks.to('cpu')
        self.num_instances, self.orig_h, self.orig_w = self.gt_masks.shape[:]
        self.click_counts += self.num_instances

        self.click_sequence = list(range(self.click_counts))
        # self.num_clicks_per_object = [[1]*self.num_instances]
        # self.num_insts = [self.num_instances]

        # if 'num_clicks_per_object' in self.inputs[0]:
        #     for x in self.inputs:
        #         self.num_clicks_per_object.append(x['num_clicks_per_object'])
        #         self.fg_coords.append(x['fg_click_coords'])
        #         self.bg_coords.append(x['bg_click_coords'])
        #         self.num_insts.append(len(x['num_clicks_per_object']))

        # self.trans_h, self.trans_w = self.inputs[0]['image'].shape[-2:]
        
        # self.ratio_h = self.trans_h/self.orig_h
        # self.ratio_w = self.trans_w/self.orig_w
        self.semantic_map = self.inputs[0]['semantic_map'].to('cpu')

        self.not_clicked_map = np.ones_like(self.gt_masks[0], dtype=np.bool)
        if self.sampling_strategy == 0:
            for coords_list in self.inputs[0]['orig_fg_click_coords']:
                for coords in coords_list:
                    self.not_clicked_map[coords[0], coords[1]] = False
        elif self.sampling_strategy == 1:
            for coords_list in self.inputs[0]['orig_fg_click_coords']:
                for coords in coords_list:
                    # self.not_clicked_map[coords[0], coords[1]] = False
                    _pm = create_circular_mask(self.orig_h, self.orig_w, centers=[[coords[0], coords[1]]], radius=self.click_radius)
                    self.not_clicked_map[np.where(_pm)] = False
        
        self.fg_orig_coords = self.inputs[0]['orig_fg_click_coords']
        self.point_coords=[]
        for j, fg_coords_per_mask in enumerate(self.fg_orig_coords):
            for i, coords in enumerate(fg_coords_per_mask):
                self.point_coords.append([coords[1], coords[0]])
        self.ignore_masks = None
        self.not_ignore_mask = None
        if 'ignore_mask' in self.inputs[0]:
            self.ignore_masks = self.inputs[0]['ignore_mask'].to(device='cpu', dtype = torch.uint8)
            self.not_ignore_mask = np.logical_not(np.asarray(self.ignore_masks, dtype=np.bool_))

    def get_next_click(self, refine_obj_index, time_step, padding=True):

        gt_mask = self.gt_masks[refine_obj_index]
        pred_mask = self.pred_masks[refine_obj_index]

        gt_mask = np.asarray(gt_mask, dtype = np.bool_)
        pred_mask = np.asarray(pred_mask, dtype = np.bool_)

        if self.not_ignore_mask is not None:
            fn_mask =  np.logical_and(np.logical_and(gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask[refine_obj_index])
            fp_mask =  np.logical_and(np.logical_and(np.logical_not(gt_mask), pred_mask), self.not_ignore_mask[refine_obj_index])
        else:
            fn_mask =  np.logical_and(gt_mask, np.logical_not(pred_mask))
            fp_mask =  np.logical_and(np.logical_not(gt_mask), pred_mask)
        
        H, W = gt_mask.shape

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist

        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

        sample_locations = [[coords_y[0], coords_x[0]]]

        obj_index = self.semantic_map[coords_y[0]][coords_x[0]] - 1
        pm = create_circular_mask(H, W, centers=sample_locations, radius=self.click_radius)
        
        if self.sampling_strategy == 0:
            self.not_clicked_map[coords_y[0], coords_x[0]] = False
        elif self.sampling_strategy == 1:
            self.not_clicked_map[np.where(pm==1)] = False

        # trans_coords = [coords_y[0]*self.ratio_h, coords_x[0]*self.ratio_w]
        if obj_index == -1:
            # if self.bg_coords[0]:
            #     # self.bg_orig_coords.extend([[coords_y[0],coords_x[0],time_step]])
            #     self.bg_coords[0].extend([[trans_coords[0],trans_coords[1],time_step]])
            # else:
            #     # self.bg_orig_coords[0] = [[coords_y[0], coords_x[0],time_step]]
            #     self.bg_coords[0] = [[trans_coords[0], trans_coords[1],time_step]]
            self.bg_orig_coords.append([coords_y[0],coords_x[0],time_step])
        else:
            # self.num_clicks_per_object[0][obj_index] += 1
            # # self.fg_orig_coords[0][obj_index].extend([[coords_y[0], coords_x[1],time_step]])
            # self.fg_coords[0][obj_index].extend([[trans_coords[0], trans_coords[1],time_step]])
            self.fg_orig_coords[obj_index].append([coords_y[0], coords_x[0],time_step])
        self.point_coords.append([coords_x[0],coords_y[0]])
        # if self.normalize_time:
        self.max_timestamps[0]+=1          

        self.click_counts+=1
        self.click_sequence.append(obj_index)
        return obj_index
    
    def compute_iou(self):

        ious = []
        if self.ignore_masks is None:
            for gt_mask, pred_mask in zip(self.gt_masks, self.pred_masks):
                intersection = (gt_mask * pred_mask).sum()
                union = torch.logical_or(gt_mask, pred_mask).to(torch.int).sum()
                ious.append(intersection/union)
            return ious
        else:
            for gt_mask, pred_mask in zip(self.gt_masks, self.pred_masks):
                ignore_gt_mask_inv = ~(self.ignore_masks[0].to(dtype=torch.bool))
                intersection = torch.logical_and(torch.logical_and(pred_mask, gt_mask), ignore_gt_mask_inv).sum()
                union = torch.logical_and(torch.logical_or(pred_mask, gt_mask), ignore_gt_mask_inv).sum()
                ious.append(intersection/union)
            return ious
    
    def compute_iou_sam(self):

        ious = []
        import itertools
        self.pred_masks = torch.from_numpy(self.pred_masks)
        if self.ignore_masks is None:
            for gt_mask, pred_mask in zip(itertools.cycle(self.gt_masks), self.pred_masks):
                intersection = (gt_mask * pred_mask).sum()
                union = torch.logical_or(gt_mask, pred_mask).to(torch.int).sum()
                ious.append(intersection/union)
            index = ious.index(max(ious))
            self.pred_masks = self.pred_masks[index][None,:,:]
            return [ious[index]]
        else:
            for gt_mask, pred_mask in zip(itertools.cycle(self.gt_masks), self.pred_masks):
                ignore_gt_mask_inv = ~(self.ignore_masks[0].to(dtype=torch.bool))
                intersection = torch.logical_and(torch.logical_and(pred_mask, gt_mask), ignore_gt_mask_inv).sum()
                union = torch.logical_and(torch.logical_or(pred_mask, gt_mask), ignore_gt_mask_inv).sum()
                ious.append(intersection/union)
            index = ious.index(max(ious))
            self.pred_masks = self.pred_masks[index][None,:,:]
            return [ious[index]]

    def save_visualization(self, save_results_path, ious=None, num_interactions=None, alpha_blend =0.6, click_radius=5):

        if num_interactions==0:
            result_masks_for_vis = self.gt_masks
        else:
            result_masks_for_vis = self.pred_masks

        image = np.asarray(self.inputs[0]['image'].permute(1,2,0))
        image = cv2.resize(image, (self.orig_w, self.orig_h))

        result_masks_for_vis = result_masks_for_vis.to(device ='cpu')
    
        pred_masks =np.asarray(result_masks_for_vis,dtype=np.uint8)
        c = []
        for i in range(pred_masks.shape[0]):
            c.append(color_map[i]/255.0)
       
        for i in range(pred_masks.shape[0]):
            image = self.apply_mask(image, pred_masks[i], c[i],alpha_blend)
        
        total_colors = len(color_map)-1
        
        point_clicks_map = np.ones_like(image)*255

        if len(self.fg_orig_coords) and num_interactions:
            for j, fg_coords_per_mask in enumerate(self.fg_orig_coords):
                for i, coords in enumerate(fg_coords_per_mask):
                    color = np.array(color_map[total_colors-5*j-4], dtype=np.uint8)
                    color = (int (color[0]), int (color[1]), int (color[2])) 
                    image = cv2.circle(image, (int(coords[1]), int(coords[0])), click_radius, tuple(color), -1)
        
        if len(self.bg_orig_coords):
            for i, coords in enumerate(self.bg_orig_coords):
                color = np.array([255,0,0], dtype=np.uint8)
                color = (int (color[0]), int (color[1]), int (color[2]))
                image = cv2.circle(image, (int(coords[1]), int(coords[0])), click_radius, tuple(color), -1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        save_dir = os.path.join(save_results_path, str(self.inputs[0]['image_id']))
        os.makedirs(save_dir, exist_ok=True)
        iou_val = np.round(sum(ious)/len(ious),4)*100
        cv2.imwrite(os.path.join(save_dir, f"tau_{num_interactions}_{iou_val}.jpg"), image)
    
    def apply_mask(self, image, mask, color, alpha=0.5):
        for c in range(3):
            image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
        return image

    def get_obj_areas(self):
        obj_areas = np.zeros(self.num_instances)
        for i in range(self.num_instances):
            obj_areas[i] = self.gt_masks[i].sum()/(self.orig_h * self.orig_w)
        return obj_areas
    
    def set_pred_masks(self,pred_masks):
        self.pred_masks = pred_masks
  