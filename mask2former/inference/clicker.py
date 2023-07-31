import torchvision
import torch
import numpy as np
import cv2
from mask2former.data.points.annotation_generator import create_circular_mask
from mask2former.evaluation.eval_utils import prepare_scribbles
from mask2former.evaluation.eval_utils import save_visualization
import os
def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j*3 + 0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3 + 1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3 + 2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))
color_map = get_palette(80)[1:]

class Clicker:

    def __init__(self, model, inputs, sampling_strategy =1, normalize_time=True, 
                click_radius = 5):
        
        self.model = model
        self.inputs = inputs
        
        self.click_radius = click_radius
        #TO DO
        self.normalize_time= normalize_time
        self.batched_max_timestamp = None
        
        # For sampling next click
        self.sampling_strategy = sampling_strategy
        self.not_clicked_map = None
        self.fg_click_map = None
        self.bg_click_map = None

        self.device = None
        self.click_counts = 0
        self.prev_mask_logits = None
        self.processed_results = None
        self.outputs = None
        self.images=None
        self.scribbles=None
        self.num_insts = None
        self.features = None
        self.mask_features = None
        self.transformer_encoder_features = None 
        self.multi_scale_features=None
        self.batched_num_scrbs_per_mask = None
        self.batched_fg_coords_list = None
        self.batched_bg_coords_list = None
        self.fg_orig_list = []
        self.bg_orig_list = []
        self._set_gt_info()
        if self.normalize_time:
            self.batched_max_timestamp = [self.num_instances-1]
    
    def _set_gt_info(self):

        self.gt_masks = self.inputs[0]['instances'].gt_masks.to('cpu')
        self.num_instances, self.orig_h, self.orig_w = self.gt_masks.shape[:]
        self.click_counts += self.num_instances
        self.trans_h, self.trans_w = self.inputs[0]['image'].shape[-2:]
        
        self.ratio_h = self.trans_h/self.orig_h
        self.ratio_w = self.trans_w/self.orig_w
        self.semantic_map = self.inputs[0]['semantic_map'].to('cpu')

        self.not_clicked_map = np.ones_like(self.gt_masks[0], dtype=np.bool)
        if self.sampling_strategy == 0:
            for coords_list in self.inputs[0]['orig_fg_click_coords']:
                for coords in coords_list:
                    self.not_clicked_map[coords[0], coords[1]] = False
        elif self.sampling_strategy == 1:
            all_scribbles = torch.cat(self.inputs[0]['fg_scrbs']).to('cpu')
            point_mask = torch.max(all_scribbles,dim=0).values
            self.not_clicked_map[torch.where(point_mask)] = False
        
        self.fg_orig_list = self.inputs[0]['orig_fg_click_coords']
        self.ignore_masks = None
        self.not_ignore_mask = None
        if 'ignore_mask' in self.inputs[0]:
            self.ignore_masks = self.inputs[0]['ignore_mask'].to(device='cpu', dtype = torch.uint8)
            self.not_ignore_mask = np.logical_not(np.asarray(self.ignore_masks, dtype=np.bool_))

    
    def predict(self):
        if self.features is None:
            (processed_results, outputs, self.images, self.scribbles,
            self.num_insts, self.features, self.mask_features,
            self.transformer_encoder_features, self.multi_scale_features,
            self.batched_num_scrbs_per_mask, self.batched_fg_coords_list,
            self.batched_bg_coords_list) = self.model(self.inputs, batched_max_timestamp=self.batched_max_timestamp)
            # self.device = self.images.tensor.device
        else:
            (processed_results, outputs, self.images, self.scribbles,
            self.num_insts, self.features, self.mask_features,
            self.transformer_encoder_features, self.multi_scale_features,
            self.batched_num_scrbs_per_mask, self.batched_fg_coords_list,
            self.batched_bg_coords_list) = self.model(self.inputs, self.images, self.scribbles, self.num_insts,
                                                self.features, self.mask_features, self.transformer_encoder_features,
                                                self.multi_scale_features, self.prev_mask_logits,
                                                self.batched_num_scrbs_per_mask,
                                                self.batched_fg_coords_list, self.batched_bg_coords_list,
                                                batched_max_timestamp = self.batched_max_timestamp)
        self.device = self.images.tensor.device
        self.pred_masks = processed_results[0]['instances'].pred_masks.to('cpu',dtype=torch.uint8)
       
        return self.compute_iou()

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

        trans_coords = [coords_y[0]*self.ratio_h, coords_x[0]*self.ratio_w]
        if obj_index == -1:
            if self.batched_bg_coords_list[0]:
                # self.bg_orig_list.extend([[coords_y[0],coords_x[0],time_step]])
                self.batched_bg_coords_list[0].extend([[trans_coords[0],trans_coords[1],time_step]])
            else:
                # self.bg_orig_list[0] = [[coords_y[0], coords_x[0],time_step]]
                self.batched_bg_coords_list[0] = [[trans_coords[0], trans_coords[1],time_step]]
            self.bg_orig_list.append([coords_y[0],coords_x[0],time_step])
        else:
            self.batched_num_scrbs_per_mask[0][obj_index] += 1
            # self.fg_orig_list[0][obj_index].extend([[coords_y[0], coords_x[1],time_step]])
            self.batched_fg_coords_list[0][obj_index].extend([[trans_coords[0], trans_coords[1],time_step]])
            self.fg_orig_list[obj_index].append([coords_y[0], coords_x[0],time_step])
        if self.normalize_time:
            self.batched_max_timestamp[0]+=1          

        self.click_counts+=1
        return obj_index
    
    def get_next_click_max_dt(self, time_step, padding=True):

        gt_masks = np.asarray(self.gt_masks, dtype = np.bool_)
        pred_masks = np.asarray(self.pred_masks, dtype = np.bool_)
        H, W = pred_masks[0].shape
        semantic_map = np.asarray(self.semantic_map)
        num_objects = pred_masks.shape[0]
        pred_semantic_map = np.zeros(pred_masks.shape[-2:], dtype=np.uint8)
        for i in range(0,num_objects):
            pred_semantic_map[pred_masks[i]==True] = i+1
        
        error_mask = pred_semantic_map!=semantic_map

        if padding:
            error_mask = np.pad(error_mask, ((1, 1), (1, 1)), 'constant')

        error_mask_dt = cv2.distanceTransform(error_mask.astype(np.uint8), cv2.DIST_L2, 0)

        if padding:
            error_mask_dt = error_mask_dt[1:-1, 1:-1]
        
        error_mask_dt = error_mask_dt * self.not_clicked_map

        _max_dist = np.max(error_mask_dt)
    
        is_positive = True
        
        coords_y, coords_x = np.where(error_mask_dt == _max_dist)  # coords is [y, x]

        sample_locations = [[coords_y[0], coords_x[0]]]

        obj_index = semantic_map[coords_y[0]][coords_x[0]] - 1
        pm = create_circular_mask(H, W, centers=sample_locations, radius=self.click_radius)
        
        if self.sampling_strategy == 0:
            self.not_clicked_map[coords_y[0], coords_x[0]] = False
        elif self.sampling_strategy == 1:
            self.not_clicked_map[np.where(pm==1)] = False
        
        trans_coords = [coords_y[0]*self.ratio_h, coords_x[0]*self.ratio_w]
        if obj_index == -1:
            if self.batched_bg_coords_list[0]:
                self.batched_bg_coords_list[0].extend([[trans_coords[0],trans_coords[1],time_step]])
            else:
                self.batched_bg_coords_list[0] = [[trans_coords[0], trans_coords[1],time_step]]
        else:
            self.batched_num_scrbs_per_mask[0][obj_index] += 1
            self.batched_fg_coords_list[0][obj_index].extend([[trans_coords[0], trans_coords[1],time_step]])

        if self.normalize_time:
            self.batched_max_timestamp[0]+=1   
        
        self.click_counts+=1
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

    def save_visualization(self, save_results_path, ious=None, num_interactions=None, alpha_blend =0.6, click_radius=3):

        # save_visualization(self.inputs[0], self.pred_masks, self.batched_fg_coords_list[0], self.batched_bg_coords_list[0],
                                # save_results_path, sum(ious)/len(ious), num_interactions,  alpha_blend=0.6)

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
            # c.append(color_map[2*(i)+2]/255.0)
            c.append(color_map[i]/255.0)
        # pred_masks = np.asarray(pred_masks).astype(np.bool_)
        # vis = visualizer.overlay_instances(masks = pred_masks, assigned_colors=c,alpha=alpha_blend)

        # [Optional] prepare labels

        # image = vis.get_image()
        for i in range(pred_masks.shape[0]):
            image = self.apply_mask(image, pred_masks[i], c[i],alpha_blend)
        # # Laminate your image!
        # fig = overlay_masks(image, masks, labels=mask_labels, colors=cmap, mask_alpha=0.5)
        total_colors = len(color_map)-1
        
        point_clicks_map = np.ones_like(image)*255
        # if not show_only_masks:
        if len(self.fg_orig_list) and num_interactions:
            for j, fg_coords_per_mask in enumerate(self.fg_orig_list):
                for i, coords in enumerate(fg_coords_per_mask):
                    color = np.array(color_map[total_colors-5*j-4], dtype=np.uint8)
                    color = (int (color[0]), int (color[1]), int (color[2])) 
                    image = cv2.circle(image, (int(coords[1]), int(coords[0])), click_radius, tuple(color), -1)
        
        if len(self.bg_orig_list):
            for i, coords in enumerate(self.bg_orig_list):
                color = np.array([255,0,0], dtype=np.uint8)
                color = (int (color[0]), int (color[1]), int (color[2]))
                image = cv2.circle(image, (int(coords[1]), int(coords[0])), click_radius, tuple(color), -1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = cv2.resize(image, (self.inputs[0]["width"],self.inputs[0]["height"]))
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
    # @property
    # def num_instances(self):
    #     return self._num_instances
